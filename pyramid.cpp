#include "pyramid.h"
#include <string>

#define PATH "D://from_ImageNet/"
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

Pyramid::Pyramid(Mat& src)
{
	int length = MIN2(src.rows, src.cols);
	int cnt = 0;
	// when one side of the src image is less than 2 * blur_radius,
	// stop computing the image on the upper level.
	while (length > 2 * KERNEL_SIZE)
	{
		length /= 2;
		cnt++;
	}
	level = cnt;
	GImages.resize(level);
	LImages.resize(level - 1);
	GImages[0] = src;  // the bottom level
}

void Pyramid::get_gaussian_pyramid()  // down-sample
{  // at this moment the pyramid only has the bottom level
	for (int i = 0; i <= level - 2; i++)
		pyrDown(GImages[i], GImages[i + 1], Size(GImages[i].cols / 2, GImages[i].rows / 2));
}

void Pyramid::get_laplace_pyramid()  // up-sample
{
	Mat temp;
	for (int i = 0; i <= level - 2; i++)
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
	if (dst_level < 0 || dst_level >= level - 1)
	{
		cout << "Out of range." << endl;
		exit(1);
	}
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
	for (int i = 0; i < level; i++)
	{
		imwrite(PATH + file1 + name1, GImages[i]);
		name1[name1.size() - 5]++;
		if (i == level - 1)
			continue;
		imwrite(PATH + file2 + name2, LImages[i]);
		name2[name2.size() - 5]++;
	}
}
