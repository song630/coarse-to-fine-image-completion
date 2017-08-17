#include "pyramid.h"
#include <string>

#define PATH "D://from_ImageNet/"

using namespace std;
using namespace cv;

Pyramid::Pyramid(const Kernel& K, Mat& src) : kernel(K)  // use copy ctor of "Kernel"
{
	int length = std::min(src.rows, src.cols);
	int cnt = 0;
	// when one side of the src image is less than 2 * blur_radius,
	// stop computing the image on the upper level.
	while (length > 2 * kernel.get_radius())
	{
		length /= 2;
		cnt++;
	}
	level = cnt;
	GImages.resize(level);
	LImages.resize(level - 1);
	GImages[0] = src;  // the bottom level
}

void Pyramid::get_gaussian_pyramid()  // call down_sample()
{  // at this moment the pyramid only has the bottom level
	for (int i = 0; i <= level - 2; i++)
		GImages[i + 1] = down_sample(i);
}

void Pyramid::get_laplace_pyramid()  // call up_sample()
{
	for (int i = 0; i <= level - 2; i++)
	{  // "-": DoG (difference of Gaussian)
		LImages[i] = GImages[i] - up_sample(i + 1);
		cv::normalize(LImages[i], LImages[i], 255, 0, CV_MINMAX);
	}
}

Mat Pyramid::get_real_image(const int dst_level)
{
	if (dst_level < 0 || dst_level >= level - 1)
	{
		cout << "Out of range." << endl;
		exit(1);
	}
	return LImages[dst_level] + up_sample(dst_level + 1);
}

void Pyramid::save_images()
{
	string name1 = "gaussian1.jpg";
	string name2 = "laplace1.jpg";
	for (int i = 0; i < level; i++)
	{
		imwrite(PATH + name1, GImages[i]);
		name1[name1.size() - 5]++;
		if (i == level - 1)
			continue;
		imwrite(PATH + name2, LImages[i]);
		name2[name2.size() - 5]++;
	}
}

// will be called in iterations
Mat Pyramid::down_sample(const int cur_level)
{
	if (GImages[cur_level].empty())
	{
		cout << "No source image." << endl;
		exit(1);
	}
	if (std::min(GImages[cur_level].rows, GImages[cur_level].cols) < kernel.get_radius() * 2)
	{
		cout << "The image is not large enough" << endl;
		exit(1);
	}
	// blur src image
	Mat temp = kernel.Gaussian_smooth(GImages[cur_level]);
	int n_row = temp.rows / 2;
	int n_col = temp.cols / 2;
	Mat up_img(n_row, n_col, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i <= n_row - 1; i++)
	{
		Vec3b *p1 = up_img.ptr<Vec3b>(i);
		Vec3b *p2 = temp.ptr<Vec3b>(2 * i + 1);  // discard even rows
		for (int j = 0; j <= n_col - 1; j++)
		{  // get odd cols
			p1[j][0] = p2[j * 2 + 1][0];  // blue
			p1[j][1] = p2[j * 2 + 1][1];  // green
			p1[j][2] = p2[j * 2 + 1][2];  // red
		}
	}
	return up_img;
}  // then the new image should be stored in "GImage[]"

string bbb = "temp1.jpg";

Mat Pyramid::up_sample(const int cur_level)
{
	if (cur_level <= 0)
	{
		cout << "Out of range." << endl;
		exit(1);
	}
	int n_row = GImages[cur_level - 1].rows;
	int n_col = GImages[cur_level - 1].cols;
	// first reserve previous pixels and fill the new pixels with 0:
	Mat down_img(n_row, n_col, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < GImages[cur_level].rows; i++)
	{
		Vec3b *p1 = GImages[cur_level].ptr<Vec3b>(i);
		Vec3b *p2 = down_img.ptr<Vec3b>(i * 2 + 1);
		for (int j = 0; j < GImages[cur_level].cols; j++)
		{  // only pixels in odd rows and cols are updated
			p2[j * 2 + 1][0] = p1[j][0];
			p2[j * 2 + 1][1] = p1[j][1];
			p2[j * 2 + 1][2] = p1[j][2];
		}
	}
	// then perform convolution using 4 * kernel:
	// kernel.normalize_mul_4();  // first * 4
	// the function Gaussian_smooth() will copy "down_img"
	Mat rst = kernel.Gaussian_smooth_4(down_img);
	cv::normalize(rst, rst, 255, 0, CV_MINMAX);
	imwrite(PATH + bbb, rst);
	bbb[4]++;
	// kernel.normalize_div_4();  // recover
	return down_img;
}
