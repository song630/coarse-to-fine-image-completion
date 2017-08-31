#include "pyramid.h"
#include <string>

#define PATH "D://from_ImageNet/"
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

Pyramid::Pyramid(const Kernel& k, const Mat& img) : K(k)
{
	int length = MIN2(img.rows, img.cols);
	int cnt = 0;
	// when one side of the src image is less than 2 * blur_radius,
	// stop computing the image on the upper level.
	/*
	while (true)
	{
		cout << "length: " << length << endl;
		length >>= 1;
		if (length <= 2 * KERNEL_SIZE)
			break;
		cnt++;
	}
	*/
	while (true)
	{
		length >>= 1;
		if (length <= 2)
			break;
		cnt++;
	}
	level = cnt;
	GImages.resize(level + 1);
	LImages.resize(level);
	GImages[0] = img;  // the bottom level
}

Pyramid::Pyramid(const Pyramid& p) : K(p.K)  // copy ctor
{
	level = p.level;
	GImages.resize(level + 1);
	LImages.resize(level);
	for (int i = 0; i <= level - 1; i++)
	{
		p.GImages[i].copyTo(GImages[i]);
		p.LImages[i].copyTo(LImages[i]);
	}
	p.GImages[level].copyTo(GImages[level]);
}

Pyramid& Pyramid::operator=(const Pyramid& p)
{
	if (this != &p)
	{
		level = p.level;
		GImages.resize(level + 1);
		LImages.resize(level);
		for (int i = 0; i <= level - 1; i++)
		{
			p.GImages[i].copyTo(GImages[i]);
			p.LImages[i].copyTo(LImages[i]);
		}
		p.GImages[level].copyTo(GImages[level]);
	}
	return *this;
}

void Pyramid::compute_gaussian_pyramid()  // down-sample
{  // at this moment the pyramid only has the bottom level
	/*  previous:
	for (int i = 0; i <= level - 1; i++)
		pyrDown(GImages[i], GImages[i + 1], Size(GImages[i].cols / 2, GImages[i].rows / 2));
	*/
	for (int i = 0; i <= level - 1; i++)
		GImages[i + 1] = down_sample(i);
}

void Pyramid::compute_laplace_pyramid()  // up-sample
{
	/*  previous:
	Mat temp;
	for (int i = 0; i <= level - 1; i++)
	{  // "-": DoG (difference of Gaussian)
		pyrUp(GImages[i + 1], temp, Size(GImages[i + 1].cols * 2, GImages[i + 1].rows * 2));
		// Mat a - b: a and b must be of the same size
		int n_rows = MIN2(GImages[i].rows, temp.rows);
		int n_cols = MIN2(GImages[i].cols, temp.cols);
		LImages[i] = GImages[i](Rect(0, 0, n_cols, n_rows)) - temp(Rect(0, 0, n_cols, n_rows));
	}
	*/
	for (int i = 0; i <= level - 1; i++)
		LImages[i] = GImages[i] - up_sample(i + 1);
}

Mat Pyramid::get_real_image(const int dst_level)
{
	if (dst_level < 0 || dst_level > level)
	{
		cout << "levels: " << level << endl;
		cout << "Out of range: " << dst_level << endl;
		exit(1);
	}
	if (dst_level == level)  // the top level
		return GImages[level];
	Mat temp;
	pyrUp(GImages[dst_level + 1], temp,
		Size(GImages[dst_level + 1].cols * 2, GImages[dst_level + 1].rows * 2));
	// Mat a + b: a and b must be of the same size
	int n_rows = MIN2(GImages[dst_level].rows, temp.rows);
	int n_cols = MIN2(GImages[dst_level].cols, temp.cols);
	return LImages[dst_level](Rect(0, 0, n_cols, n_rows)) + temp(Rect(0, 0, n_cols, n_rows));
}

void Pyramid::save_images()
{
	string name1 = "gaussian1.jpg";
	string name2 = "laplace1.jpg";
	string file1 = "gaussian/";
	string file2 = "laplace/";
	for (int i = 0; i <= level; i++)
	{
		imwrite(PATH + file1 + name1, GImages[i]);
		name1[name1.size() - 5]++;
		if (i == level)
			continue;
		imwrite(PATH + file2 + name2, LImages[i]);
		name2[name2.size() - 5]++;
	}
}

// will be called in iterations
Mat Pyramid::down_sample(const int cur_level)
{
	if (GImages[cur_level].empty())
	{
		cout << "No source image." << endl;
		system("pause");
		exit(1);
	}
	if (std::min(GImages[cur_level].rows, GImages[cur_level].cols) < K.get_radius() * 2)
	{
		cout << "The image is not large enough" << endl;
		system("pause");
		exit(1);
	}
	// blur src image
	Mat temp = K.Gaussian_smooth(GImages[cur_level]);
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

Mat Pyramid::up_sample(const int cur_level)  // to get laplace images
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
	for (int i = 0; i < n_row; i++)
	{
		Vec3b *p2 = down_img.ptr<Vec3b>(i);  // new
		Vec3b *p1 = GImages[cur_level].ptr<Vec3b>(i / 2);  // previous
		for (int j = 0; j < n_col; j++)
		{  // only pixels in odd rows and cols are updated
			p2[j][0] = p1[j / 2][0];
			p2[j][1] = p1[j / 2][1];
			p2[j][2] = p1[j / 2][2];
		}
	}
	return down_img;
}
