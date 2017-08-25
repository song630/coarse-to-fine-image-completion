#include "pyramid.h"
#include <string>

#define PATH "D://from_ImageNet/"
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

Pyramid::Pyramid(const Mat& img)
{
	int length = MIN2(img.rows, img.cols);
	int cnt = 0;
	// when one side of the src image is less than 2 * blur_radius,
	// stop computing the image on the upper level.
	while (true)
	{
		cout << "length: " << length << endl;
		length >>= 1;
		if (length <= 2 * KERNEL_SIZE)
			break;
		cnt++;
	}
	level = cnt;
	GImages.resize(level + 1);
	LImages.resize(level);
	GImages[0] = img;  // the bottom level
}

Pyramid::Pyramid(const Pyramid& p)  // copy ctor
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
	for (int i = 0; i <= level - 1; i++)
		pyrDown(GImages[i], GImages[i + 1], Size(GImages[i].cols / 2, GImages[i].rows / 2));
}

void Pyramid::compute_laplace_pyramid()  // up-sample
{
	Mat temp;
	for (int i = 0; i <= level - 1; i++)
	{  // "-": DoG (difference of Gaussian)
		pyrUp(GImages[i + 1], temp, Size(GImages[i + 1].cols * 2, GImages[i + 1].rows * 2));
		// Mat a - b: a and b must be of the same size
		int n_rows = MIN2(GImages[i].rows, temp.rows);
		int n_cols = MIN2(GImages[i].cols, temp.cols);
		LImages[i] = GImages[i](Rect(0, 0, n_cols, n_rows)) - temp(Rect(0, 0, n_cols, n_rows));
	}
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
