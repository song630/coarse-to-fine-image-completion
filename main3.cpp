#include "pyramid.h"
#include "segmentation.h"
#include "patchmatch.h"
#include "completion.h"
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/src2.jpg");
	completion solution(src);
	solution.initialize();
	Mat rst = solution.image_complete();
	namedWindow("Completed Image");
	imshow("Completed Image", rst);
	imwrite("D://from_ImageNet/completed.jpg", rst);
	waitKey(0);

	return 0;
}
