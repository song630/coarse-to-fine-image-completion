/* kernel.h
 * Gaussian kernel, used for building Gaussian pyramid and Laplace pyramid
 */

#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "segmentation.h"  // use RECT
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;
using namespace cv;

enum KERNEL_TYPE {
	gaussian_3 = 3,
	gaussian_5 = 5,
	gaussian_7 = 7
};

// sigma = 0.84089642
static const double gaussian_kernel_7[7][7] =
{
	{ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067 },
	{ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292 },
	{ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117 },
	{ 0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771 },
	{ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117 },
	{ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292 },
	{ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067 }
};

static const double gaussian_kernel_5[5][5] =
{
	{ 0.00000659, 0.00042478, 0.00170354, 0.00042478, 0.00000659 },
	{ 0.00042478, 0.02739840, 0.10987800, 0.02739840, 0.00042478 },
	{ 0.00170354, 0.10987800, 0.44065500, 0.10987800, 0.00170354 },
	{ 0.00042478, 0.02739840, 0.10987800, 0.02739840, 0.00042478 },
	{ 0.00000659, 0.00042478, 0.00170354, 0.00042478, 0.00000659 }
};

static const double gaussian_kernel_3[3][3] =
{
	{ 0.00001472, 0.00380683, 0.00001472 },
	{ 0.00380683, 0.98471400, 0.00380683 },
	{ 0.00001472, 0.00380683, 0.00001472 }
};

class Kernel {  // Gaussian kernel
public:
	// appoint a "blur_radius" and "sigma"
	Kernel(const int r, const float s) : blur_radius(r), sigma(s) {  // ctor1
		if (r <= 0 || r % 2 != 1)
		{
			cout << "Invalid input of blur radius." << endl;
			exit(1);
		}
		mask.resize(r);  // initialize the vector "mask"
		KernelMask::iterator iter;
		for (iter = mask.begin(); iter != mask.end(); iter++)
			(*iter).resize(r);  // r * r kernel
		compute_normalize();
	}

	// use accessible kernels
	// notice: if this ctor has been employed, there is no need calling compute_normalize()
	Kernel(const KERNEL_TYPE type);  // ctor2

	Kernel(const Kernel& K);

	~Kernel() {
		KernelMask::iterator iter;
		for (iter = mask.begin(); iter != mask.end(); iter++)
			(*iter).clear();
		mask.clear();
	}

	Mat Gaussian_smooth(const Mat& src);
	Mat Gaussian_smooth(const Mat& src, const RECT& roi);  // do not compute inside the hole
	void print();
	int get_radius() {
		return blur_radius;
	}

private:
	// 2-d array, stores the weights of every pixel in matrix
	typedef vector<vector<float> > KernelMask;
	KernelMask mask;
	const float sigma;
	// according to http://amitapba.blog.163.com/blog/static/203610207201281992239/,
	// the program should compute a matrix whose side is of (6 * sigma + 1).
	const int blur_radius;  // i.e. 3 * 3 or 5 * 5

	void compute_normalize();  // compute the mask
};

#endif
