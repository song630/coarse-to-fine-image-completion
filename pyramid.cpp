#include "pyramid.h"
#include <string>

#define PATH "D://from_ImageNet/"
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

Pyramid::Pyramid(const Kernel& k, const Mat& masked, const Mat& src) : K(k)  // img: masked
{
	int length = MIN2(masked.rows, masked.cols);
	int cnt = 0;
	GImages.push_back(masked);  // the bottom level
	SrcImages.push_back(src);
}

Pyramid::Pyramid(const Pyramid& p) : K(p.K)  // copy ctor
{
	level = p.level;
	GImages.resize(level + 1);
	SrcImages.resize(level + 1);
	LImages.resize(level);
	for (int i = 0; i <= level - 1; i++)
	{
		p.GImages[i].copyTo(GImages[i]);
		p.LImages[i].copyTo(LImages[i]);
		p.SrcImages[i].copyTo(SrcImages[i]);
	}
	p.GImages[level].copyTo(GImages[level]);
	p.SrcImages[level].copyTo(SrcImages[level]);
}

Pyramid& Pyramid::operator=(const Pyramid& p)
{
	if (this != &p)
	{
		level = p.level;
		GImages.resize(level + 1);
		SrcImages.resize(level + 1);
		LImages.resize(level);
		for (int i = 0; i <= level - 1; i++)
		{
			p.GImages[i].copyTo(GImages[i]);
			p.LImages[i].copyTo(LImages[i]);
			p.SrcImages[i].copyTo(SrcImages[i]);
		}
		p.GImages[level].copyTo(GImages[level]);
		p.SrcImages[level].copyTo(SrcImages[level]);
	}
	return *this;
}

void Pyramid::compute_src_pyramid(const vector<RECT>& roi_vec)
{  // without hole, used for up_sample()
	level = roi_vec.size();  // "level" decided here
	for (int i = 0; i <= level - 1; i++)
		SrcImages.push_back(down_sample(i, roi_vec[i], SrcImage));
}

void Pyramid::compute_gaussian_pyramid(const vector<RECT>& roi_vec)  // down-sample
{  // at this moment the pyramid only has the bottom level
	// level = roi_vec.size();  // "level" decided here
	for (int i = 0; i <= level - 1; i++)
		GImages.push_back(down_sample(i, roi_vec[i], GImage));
}

void Pyramid::compute_laplace_pyramid(const vector<RECT>& roi_vec)  // up-sample
{
	LImages.resize(level);
	for (int i = 0; i <= level - 1; i++)
		LImages[i] = GImages[i] - up_sample(i + 1, roi_vec[i]);
}

Mat Pyramid::get_real_image(const int dst_level, const RECT& roi)
{
	if (dst_level < 0 || dst_level > level)
	{
		cout << "levels: " << level << endl;
		cout << "Out of range: " << dst_level << endl;
		exit(1);
	}
	if (dst_level == level)  // the top level
		return GImages[level];
	Mat temp = up_sample(dst_level + 1, roi);
	return LImages[dst_level] + temp;
}

void Pyramid::save_images()
{
	string name1 = "gaussian1.jpg";
	string name2 = "laplace1.jpg";
	string name5 = "src1.jpg";
	string file1 = "gaussian/";
	string file2 = "laplace/";
	string file3 = "src/";
	for (int i = 0; i <= level; i++)
	{
		imwrite(PATH + file1 + name1, GImages[i]);
		name1[name1.size() - 5]++;
		imwrite(PATH + file3 + name5, SrcImages[i]);
		name5[3]++;
		if (i == level)
			continue;
		imwrite(PATH + file2 + name2, LImages[i]);
		name2[name2.size() - 5]++;
	}
}

// will be called in iterations
Mat Pyramid::down_sample(const int cur_level, const RECT& roi, const PYR_TYPE& type)
{
	GPyramid Images;
	if (type == GImage)
		Images = GImages;
	else
		Images = SrcImages;
	if ((Images)[cur_level].empty())
	{
		cout << "No source image." << endl;
		system("pause");
		exit(1);
	}
	if (std::min((Images)[cur_level].rows, (Images)[cur_level].cols) < K.get_radius() * 2)
	{
		cout << "The image is not large enough" << endl;
		system("pause");
		exit(1);
	}
	// blur src image
	Mat temp = K.Gaussian_smooth((Images)[cur_level], roi);
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

Mat Pyramid::up_sample(const int cur_level, const RECT& roi)  // to get laplace images
{
	if (cur_level <= 0)
	{
		cout << "Out of range." << endl;
		exit(1);
	}
	int n_row = SrcImages[cur_level - 1].rows;
	int n_col = SrcImages[cur_level - 1].cols;
	// first reserve previous pixels and fill the new pixels with 0:
	Mat down_img(n_row, n_col, CV_8UC3, Scalar(0, 0, 0));
		
	for (int i = 0; i <= n_row - 1; i++)
	{
		Vec3b *p2 = down_img.ptr<Vec3b>(i);  // new
		int dst_row = i / 2;
		if (i == n_row - 1)  // the last row
			dst_row = ((i / 2) == SrcImages[cur_level].rows) ? i / 2 - 1 : i / 2;
		Vec3b *p1 = SrcImages[cur_level].ptr<Vec3b>(dst_row);  // previous
		for (int j = 0; j <= n_col - 1; j++)
		{  // only pixels in odd rows and cols are updated
			int dst_col = j / 2;
			if (j == n_col - 1)  // the last column
				dst_col = ((j / 2) == SrcImages[cur_level].cols) ? j / 2 - 1 : j / 2;
			p2[j][0] = p1[dst_col][0];
			p2[j][1] = p1[dst_col][1];
			p2[j][2] = p1[dst_col][2];
		}
	}
	// set the color of all pixels within the hole 0:
	for (int i = roi.first.y; i <= roi.second.y; i++)
	{
		Vec3b *p = down_img.ptr<Vec3b>(i);
		for (int j = roi.first.x; j <= roi.second.x; j++)
		{
			p[j][0] = 0;
			p[j][1] = 0;
			p[j][2] = 0;
		}
	}
	return down_img;
}
