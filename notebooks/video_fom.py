#!/usr/bin/env python
#
# @copyright: AlertAvert.com (c) . All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = 'Marco Massenzio'
__email__ = 'marco@alertavert.com'

import argparse
import logging
import math
import os

import cv2
import halo
import numpy as np

LOG_FORMAT = '%(asctime)s [%(levelname)-5s] %(message)s'

# Note: Codec/file extension are a "delicate" combination with OpenCV.
# I have found the following codecs to work in Ubuntu 17.10, writing to an .mp4 file.
# Change the below at your own risk.
CODECS = ['MP4V', 'H264', 'MJPG', 'XVID', 'X264']
EXT = '.mp4'


def parse_args():
    """ Parse command line arguments and returns a configuration object.

    @return: the configuration object, arguments accessed via dotted notation
    @rtype: Namespace
    """
    parser = argparse.ArgumentParser(description="Processes a video stream, detecting fast-moving "
                                                 "objects (FOM) and displaying a box in each "
                                                 "frame one is detected.")

    parser.add_argument('--logdir', default=None,
                        help="The direcory to use for the log files, if not set, uses stdout")

    parser.add_argument('--debug', '-v', default=False, action='store_true')

    parser.add_argument('-o', dest='outfile',
                        help="An optional output file for the video; otherwise the processed "
                             "video will be played in a window in real time. OMIT the extension.")

    parser.add_argument('--codec', choices=CODECS, default='MP4V',
                        help="The chosen codec to encode the video, by default MP4")

    parser.add_argument('infile', help="The video file to process for FOM detection")
    return parser.parse_args()


def detect_fom(frames, **kwargs):
    """ Detects a FOM in the first frame of `frames` and returns a bounding rectangle, if found.

        `kwargs` should contain the following configuration options (or default values will be used)

            - `threshold` to convert a grayscale image to a binary (b/w) one;
            - `min_area` to eliminate from the "FOM candidates" small details and noise;
            - `psi` a "thinning threshold," to convert a wide FOM trace to a linear path;
            - `gamma` an area matching threshold, part of the "motion model;"

        For more details on the above options see:
            [Rozumnyi Kotera Sroubek Novotny Matas, CVPR 2017, "The World of Fast Moving Objects"]
            (https://arxiv.org/pdf/1611.07889.pdf)

        :param frames: a 3-tuple of frames: current, previous and next
        :type frames: tuple

        :param kwargs: configuration arguments for the detection algorithm.
        :type kwargs: dict
    """
    im_t, im_tm1, im_tp1 = frames
    delta_plus = cv2.absdiff(im_t, im_tm1)
    delta_0 = cv2.absdiff(im_tp1, im_tm1)
    delta_minus = cv2.absdiff(im_t,im_tp1)

    th = kwargs.get('threshold', 10.0)
    _, dbp = cv2.threshold(delta_plus, th, 255, cv2.THRESH_BINARY)
    _, dbm = cv2.threshold(delta_minus, th, 255, cv2.THRESH_BINARY)
    _, db0 = cv2.threshold(delta_0, th, 255, cv2.THRESH_BINARY)

    # Motion detection image - it should be negated, but we have already have the negative
    # (due to the way the threshold() method works - see the Jupyter Notebook for an example.
    detect = cv2.bitwise_and(cv2.bitwise_and(dbp, dbm), cv2.bitwise_not(db0))

    # We find here the "connected components" to identify FOM candidates.
    _, _, stats, _ = cv2.connectedComponentsWithStats(detect, ltype=cv2.CV_16U)
    min_area = kwargs.get('min_area', 500)
    candidates = list()
    for stat in stats:
        area = stat[cv2.CC_STAT_AREA]
        if area < min_area:
            continue  # Skip small objects (noise)

        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
        candidates.append((lt, rb, area))

    # Thinning threshold.
    psi = kwargs.get('psi', 0.7)

    # Area matching threshold.
    gamma = kwargs.get('gamma', 0.3)

    for candidate in candidates:
        # The first two elements of each `candidate` tuple are
        # the opposing corners of the bounding box.
        x1, y1 = candidate[0]
        x2, y2 = candidate[1]

        # We had placed the candidate's area in the third element of the tuple.
        actual_area = candidate[2]

        # For each candidate, estimate the "radius" using a distance transform.
        # The transform is computed on the (small) bounding rectangle.
        cand = detect[y1:y2, x1:x2]
        dt = cv2.distanceTransform(cand, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        radius = np.amax(dt)
        if radius == 0:
            continue

        # "Thinning" of pixels "close to the center" to estimate a
        # potential FOM path.
        # TODO: compute actual path length, using best-fit straight line
        #   along the "thinned" path.
        _, Pt = cv2.threshold(dt, psi * radius, 255, cv2.THRESH_BINARY)

        # For now, we estimate it as the max possible lenght in the bounding box, its diagonal.
        w = x2 - x1
        h = y2 - y1
        path_len = math.sqrt(w * w + h * h)
        expected_area = radius * (2 * path_len + math.pi * radius)

        area_ratio = abs(actual_area / expected_area - 1)
        if area_ratio < gamma:
            return candidate


def video_frames(video):
    cap = cv2.VideoCapture(video)
    _, ff = cap.read()
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield gray
    cap.release()


def main(options):
    if not os.path.exists(options.infile):
        logging.error("Cannot open video file %s", options.infile)
        exit(1)

    writeout = False
    if options.outfile:
        writeout = True
        outfile = os.path.splitext(options.outfile)[0] + EXT
        fourcc = cv2.VideoWriter_fourcc(*options.codec)
        out_vid = cv2.VideoWriter(outfile, fourcc, 30.0, (1280, 720), False)

    logging.info("Detecting FOMs in {infile} and writing out to {out} (using: {codec})".format(
        infile=options.infile,
        out=outfile if options.outfile else "live stream",
        codec=options.codec))

    current = nxt = None
    frame_count = 0
    spinner = halo.Halo(text="Detecting FOM...", spinner='dots')
    spinner.start()
    try:
        for frame in video_frames(options.infile):
            prev = current
            current = nxt
            nxt = frame
            if prev is None or current is None:
                continue
            bounding_rect = detect_fom((current, prev, nxt))
            if bounding_rect:
                spinner.text = "Found another FOM"
                cv2.rectangle(current, bounding_rect[0], bounding_rect[1], 127, 2)

            cv2.imshow('processed', current)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if writeout:
                out_vid.write(current)

            frame_count += 1
            if frame_count % 48 == 0:
                spinner.text = "{} Frames processed".format(frame_count)
        spinner.succeed(text="Done")
    except KeyboardInterrupt:
        spinner.succeed(text="Interrupted by user")
    except Exception as ex:
        spinner.fail("An error occurred: {}".format(ex))
    finally:
        if writeout:
            out_vid.release()


if __name__ == '__main__':
    config = parse_args()
    logfile = None
    if config.logdir:
        logfile = os.path.join(os.path.expanduser(config.logdir), 'messages.log')
    level = logging.INFO
    if config.debug:
        level = logging.DEBUG
    if logfile:
        print("All logging going to {} (debug info {})".format(
            logfile, 'enabled' if config.debug else 'disabled'))

    logging.basicConfig(filename=logfile, level=level, format=LOG_FORMAT,
                        datefmt="%Y-%m-%d %H:%M:%S")
    main(config)
