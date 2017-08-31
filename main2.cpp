#include "segmentation.h"
#include "kernel.h"
#include "pyramid.h"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/src2.jpg");

	//  [test segmentation.h]
	segmentation s(src);
	s.draw_rect();
	s.print();
	Mat rst = s.get_masked();

	Kernel k(5, 0.8);
	Pyramid pyr(k, rst);
	pyr.compute_gaussian_pyramid();
	pyr.compute_laplace_pyramid();
	pyr.save_images();

	/*  [test pyramid.h]
	Pyramid P(src);
	P.compute_gaussian_pyramid();
	P.compute_laplace_pyramid();
	Mat real1 = P.get_real_image(0);
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", real1);
	waitKey(0);
	*/

	system("pause");
	return 0;
}
