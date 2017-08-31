#include "kernel.h"
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
	Kernel k(5, 0.8);
	completion solution(src, k);
	solution.initialize();
	Mat rst = solution.image_complete();
	namedWindow("Completed Image");
	imshow("Completed Image", rst);
	imwrite("D://from_ImageNet/completed.jpg", rst);
	waitKey(0);

	system("pause");
	return 0;
}
