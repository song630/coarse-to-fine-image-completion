#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;
using namespace cv;

int main(void)
{
	Mat src;
	src = imread("D://from_ImageNet/marble1.jpg");
	cout << src.rows << ", " << src.cols << endl;
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", src);
	waitKey(0);

	system("pause");
	return 0;
}
