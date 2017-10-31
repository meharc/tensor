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

import unittest

import cv2
import os

from video_fom import detect_fom


class TestVideoFom(unittest.TestCase):
    def setUp(self):
        frame = "tennis-frame-{}.png"
        self.frames = tuple(
            cv2.imread(os.path.join("data", "tennis", frame.format(n)), cv2.IMREAD_GRAYSCALE)
                for n in [4, 3, 5])

    def test_fom(self):
        bound_rect = detect_fom(self.frames)
        self.assertIsNotNone(bound_rect)
        self.assertTupleEqual(((319, 315), (442, 348), 1877), bound_rect)
