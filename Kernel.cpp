#include "kernel.h"
#include <cmath>

#define _PI 3.14159265

using namespace std;
using namespace cv;

Kernel::Kernel(const KERNEL_TYPE type) : blur_radius(type), sigma(0.84089642f) 
{
	mask.resize(type);
	KernelMask::iterator iter;
	for (iter = mask.begin(); iter != mask.end(); iter++)
		(*iter).resize(type);
	if (type == gaussian_3)
	{
		for (int i = 0; i <= blur_radius - 1; i++)
		for (int j = 0; j <= blur_radius - 1; j++)
			mask[i][j] = gaussian_kernel_3[i][j];
	}
	else if (type == gaussian_5)
	{
		for (int i = 0; i <= blur_radius - 1; i++)
		for (int j = 0; j <= blur_radius - 1; j++)
			mask[i][j] = gaussian_kernel_5[i][j];
	}
	else if (type == gaussian_7)
	{
		for (int i = 0; i <= blur_radius - 1; i++)
		for (int j = 0; j <= blur_radius - 1; j++)
			mask[i][j] = gaussian_kernel_7[i][j];
	}
	else
	{
		cout << "Illigal input." << endl;
		system("pause");
		exit(1);
	}
}

Kernel::Kernel(const Kernel& K) : sigma(K.sigma), blur_radius(K.blur_radius)  // copy ctor
{
	mask.resize(blur_radius);
	for (int i = 0; i < blur_radius; i++)
		for (int j = 0; j < blur_radius; j++)
			mask[i].push_back(K.mask[i][j]);
}

void Kernel::compute_normalize()  // compute the mask
{
	float sum = 0;  // used for normalization
	float sigma_sq = sigma * sigma;  // sigma ^ 2
	float fract = 1.f / (2 * _PI * sigma_sq);  // the constant fraction
	int center = blur_radius / 2;  // e.g. 5->2, 7->3
	for (int i = -center; i <= center; i++)
	{
		for (int j = -center; j <= center; j++)
		{
			// according to normal distribution:
			mask[i + center][j + center] = fract * exp(-1.f * (i * i + j * j) / (2 * sigma_sq));
			sum += mask[i + center][j + center];
		}
	}
	for (int i = 0; i < blur_radius; i++)  // normalize, the sum of weights must be 1.
		for (int j = 0; j < blur_radius; j++)
			mask[i][j] /= sum;
		cout << "normalization done." << endl;
}

Mat Kernel::Gaussian_smooth(const Mat& src)  // gaussian blur
{
	Mat image;
	if (src.rows <= blur_radius || src.cols <= blur_radius)
	{  // image is smaller than kernel
		cout << "Cannot blur this image." << endl;
		system("pause");
		exit(1);
	}
	src.copyTo(image);
	int center = blur_radius / 2;
	float b_value = 0, g_value = 0, r_value = 0;
	// will leave a frame of width "blur_radius"
	for (int i = 0; i <= image.rows - 1; ++i)
	{
		Vec3b *p1 = image.ptr<Vec3b>(i);  // get the first pixel of row i
		for (int j = 0; j <= image.cols - 1; ++j)
		{
			r_value = 0;
			g_value = 0;
			b_value = 0;
			for (int m = -center; m <= center; m++)  // m: kernel_row
			{
				int kernel_row = i + m;
				int kernel_column;
				// deal with the boundary: REFLECTION
				if (i + m < 0)  // below row 0
					kernel_row = -kernel_row;
				else if (i + m > image.rows - 1)  // above row max
					kernel_row = image.rows - 1 - (kernel_row - image.rows);
				Vec3b *p2 = image.ptr<Vec3b>(kernel_row);
				for (int n = -center; n <= center; n++)  // n: kernel_column
				{
					kernel_column = j + n;
					// deal with the boundary: REFLECTION
					if (j + n < 0)  // left of col 0
						kernel_column = -kernel_column;
					else if (j + n > image.cols - 1)  // right of col max
						kernel_column = image.cols - 1 - (kernel_column - image.cols);
					b_value += p2[kernel_column][0] * mask[m + center][n + center];  // blue
					g_value += p2[kernel_column][1] * mask[m + center][n + center];  // green
					r_value += p2[kernel_column][2] * mask[m + center][n + center];  // red
				}
			}
			p1[j][0] = static_cast<unsigned char>(b_value);
			p1[j][1] = static_cast<unsigned char>(g_value);
			p1[j][2] = static_cast<unsigned char>(r_value);
		}
	}
	return image;
}

Mat Kernel::Gaussian_smooth(const Mat& src, const RECT& roi)  // do not compute when inside the hole
{
	Mat image;
	if (src.rows <= blur_radius || src.cols <= blur_radius)
	{  // image is smaller than kernel
		cout << "Cannot blur this image." << endl;
		system("pause");
		exit(1);
	}
	src.copyTo(image);
	int center = blur_radius / 2;  // e.g. 5->2
	float b_value = 0, g_value = 0, r_value = 0;
	vector<vector<float> > temp_mask;  // does not include pixels inside the hole
	temp_mask.resize(blur_radius);
	for (int i = 0; i < blur_radius; i++)
		temp_mask[i].resize(blur_radius);

	for (int i = 0; i <= image.rows - 1; ++i)  // traverse the whole pic
	{
		Vec3b *p1 = image.ptr<Vec3b>(i);  // get the first pixel of row i
		for (int j = 0; j <= image.cols - 1; ++j)
		{
			if (i >= roi.first.y && i <= roi.second.y && j >= roi.first.x && j <= roi.second.x)
				continue;  // pixels inside the hole

			r_value = 0;
			g_value = 0;
			b_value = 0;
			if (i < roi.first.y - center || j < roi.first.x - center ||
				i > roi.second.y + center || j > roi.second.x + center)
			{  // will not be influenced by the hole
				for (int m = -center; m <= center; m++)  // m: kernel_row
				{
					int kernel_row = i + m;
					int kernel_column;
					// deal with the boundary: REFLECTION
					if (i + m < 0)  // below row 0
						kernel_row = -kernel_row;
					else if (i + m > image.rows - 1)  // above row max
						kernel_row = image.rows - 1 - (kernel_row - image.rows);
					Vec3b *p2 = image.ptr<Vec3b>(kernel_row);
					for (int n = -center; n <= center; n++)  // n: kernel_column
					{
						kernel_column = j + n;
						// deal with the boundary: REFLECTION
						if (j + n < 0)  // left of col 0
							kernel_column = -kernel_column;
						else if (j + n > image.cols - 1)  // right of col max
							kernel_column = image.cols - 1 - (kernel_column - image.cols);
						b_value += p2[kernel_column][0] * mask[m + center][n + center];  // blue
						g_value += p2[kernel_column][1] * mask[m + center][n + center];  // green
						r_value += p2[kernel_column][2] * mask[m + center][n + center];  // red
					}
				}
				p1[j][0] = static_cast<unsigned char>(b_value);
				p1[j][1] = static_cast<unsigned char>(g_value);
				p1[j][2] = static_cast<unsigned char>(r_value);
			}  // end-if
			else  // the kernel has some pixels inside the hole, which should be ignored
			{
				// first normalize:
				float sum = 0.f;
				for (int m = -center; m <= center; m++)  // m: kernel_row
				{
					int kernel_row = i + m;
					int kernel_column;
					// deal with the boundary: REFLECTION
					if (i + m < 0)  // below row 0
						kernel_row = -kernel_row;
					else if (i + m > image.rows - 1)  // above row max
						kernel_row = image.rows - 1 - (kernel_row - image.rows);
					for (int n = -center; n <= center; n++)  // n: kernel_column
					{
						kernel_column = j + n;
						if (kernel_row >= roi.first.y && kernel_row <= roi.second.y &&
							kernel_column >= roi.first.x && kernel_column <= roi.second.x)
						{  // the pixel within the hole
							temp_mask[m + center][n + center] = 0;
						}
						else
						{
							temp_mask[m + center][n + center] = mask[m + center][n + center];
							sum += temp_mask[m + center][n + center];
						}
					}
				}
				for (int m = 0; m < blur_radius; m++)  // normalize, the sum of weights must be 1.
				for (int n = 0; n < blur_radius; n++)
					temp_mask[m][n] /= sum;  // normalization done.

				// then compute the color:
				for (int m = -center; m <= center; m++)  // m: kernel_row
				{
					int kernel_row = i + m;
					int kernel_column;
					// deal with the boundary: REFLECTION
					if (i + m < 0)  // below row 0
						kernel_row = -kernel_row;
					else if (i + m > image.rows - 1)  // above row max
						kernel_row = image.rows - 1 - (kernel_row - image.rows);
					Vec3b *p2 = image.ptr<Vec3b>(kernel_row);
					for (int n = -center; n <= center; n++)  // n: kernel_column
					{
						kernel_column = j + n;
						// deal with the boundary: REFLECTION
						if (j + n < 0)  // left of col 0
							kernel_column = -kernel_column;
						else if (j + n > image.cols - 1)  // right of col max
							kernel_column = image.cols - 1 - (kernel_column - image.cols);
						if (temp_mask[m + center][n + center] == 0)  // pass
							continue;
						b_value += p2[kernel_column][0] * temp_mask[m + center][n + center];  // blue
						g_value += p2[kernel_column][1] * temp_mask[m + center][n + center];  // green
						r_value += p2[kernel_column][2] * temp_mask[m + center][n + center];  // red
					}
				}
				p1[j][0] = static_cast<unsigned char>(b_value);
				p1[j][1] = static_cast<unsigned char>(g_value);
				p1[j][2] = static_cast<unsigned char>(r_value);
			}  // end-else
		}
	}
	return image;
}

void Kernel::print()  // print a matrix
{
	float sum = 0;
	for (int i = 0; i < blur_radius; i++)
	{
		for (int j = 0; j < blur_radius; j++)
		{
			cout << mask[i][j] << "  ";
			sum += mask[i][j];
		}
		cout << endl;
	}
	cout << "sum = " << sum << endl;
}
