#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;

namespace {

void createLookupTable(int scale, uchar table[])
{
  for (int i = 0; i < 256; ++i) {
    table[i] = (uchar) (scale * (i / scale));
  }
}


Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
  // accept only char type matrices
  CV_Assert(I.depth() == CV_8U);
  const int channels = I.channels();
  switch(channels)
  {
    case 1:
    {
      MatIterator_<uchar> it, end;
      for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
        *it = table[*it];
      break;
    }
    case 3:
    {
      MatIterator_<Vec3b> it, end;
      for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
      {
        (*it)[0] = table[(*it)[0]];
        (*it)[1] = table[(*it)[1]];
        (*it)[2] = table[(*it)[2]];
      }
    }
  }
  return I;
}
}


int main(int argc, char **argv) {

  if (argc != 2) {
    printf("usage: DisplayImage <Image_Path>\n");
    return EXIT_FAILURE;
  }

  Mat image;
  int scale = 4;
  uchar lookupTable[256];

  double start = (double)getTickCount();
  double minVal, maxVal;

  ::createLookupTable(scale, lookupTable);
  image = imread(argv[1], 1);

  if (!image.data) {
    printf("No image data \n");
    return EXIT_FAILURE;
  }

  std::cout << image.rows << ", " << image.cols
            << "; channels: " << image.channels()
            << std::endl;


  imshow("Original Image", image);

  // Image reduction:
  Mat &detail = ::ScanImageAndReduceIterator(image, lookupTable);
  imshow("Reduced Image", detail);

  // Image conversion:
  Mat grey, draw, sobelx;
  Rect r(10, 10, 100, 100);

  cvtColor(image, grey, COLOR_BGR2GRAY);

  Sobel(grey, sobelx, CV_32F, 1, 0);
  minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities

  Mat smallImg = sobelx(r);
  smallImg = Scalar((minVal + maxVal) / 2);

  assert(maxVal - minVal > 0);

  sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
  imshow(":: conversion ::", draw);


  double elapsed = ((double)getTickCount() - start)/getTickFrequency();
  std::cout << "Time passed in seconds: " << elapsed << std::endl;

  waitKey(0);
  return EXIT_SUCCESS;
}
