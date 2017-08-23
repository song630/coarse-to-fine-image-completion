#include "segmentation.h"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/src2.jpg");

	//  [test segmentation.h]
	segmentation s(src);
	s.get_rect();
	s.print();
	Mat rst = s.get_masked();
	namedWindow("masked");
	imshow("masked", rst);
	waitKey(0);

	/*  [test pyramid.h]
	Pyramid P(src);
	P.get_gaussian_pyramid();
	P.get_laplace_pyramid();
	Mat real1 = P.get_real_image(0);
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", real1);
	waitKey(0);
	*/

	system("pause");
	return 0;
}
