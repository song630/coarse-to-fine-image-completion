#include "kernel.h"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src, dst;
	src = imread("D://from_ImageNet/test1.jpg");

	// KERNEL_TYPE type = gaussian_5;
	// Kernel k(7, 0.84089642);  // init
	Kernel k(7, 5);
	k.compute_normalize();
	dst = k.Gaussian_smooth(src);
	k.print();

	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", dst);
	waitKey(0);

	system("pause");
	return 0;
}
