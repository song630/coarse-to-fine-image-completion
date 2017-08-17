#include "kernel.h"
#include "pyramid.h"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/test1.jpg");

	Kernel K(7, 1);
	K.compute_normalize();
	K.print();

	Pyramid P(K, src);
	P.get_gaussian_pyramid();
	P.get_laplace_pyramid();
	P.save_images();

	/*
	Mat aaa = P.get_real_image(1);
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", aaa);
	waitKey(0);
	*/
	/*
	src = imread("D://from_ImageNet/gaussian3.jpg");
	Mat dst;
	pyrUp(src, dst, Size(src.cols * 2, src.rows * 2));
	imwrite("D://from_ImageNet/up.jpg", dst);
	src = imread("D://from_ImageNet/gaussian2.jpg");
	cout << dst.rows << " " << dst.cols << endl;
	cout << src.rows << " " << src.cols << endl;
	Mat sub = src(Rect(0, 0, 414, 390)) - dst;
	imwrite("D://from_ImageNet/sub.jpg", sub);
	*/

	system("pause");
	return 0;
}
