#include "pyramid.h"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/test1.jpg");

	Pyramid P(src);
	P.get_gaussian_pyramid();
	P.get_laplace_pyramid();
	// P.save_images();

	Mat real1 = P.get_real_image(0);
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", real1);
	waitKey(0);

	system("pause");
	return 0;
}
