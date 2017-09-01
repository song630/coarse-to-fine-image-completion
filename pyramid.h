/* pyramid.h
 * both Gaussian and Laplace pyramid included.
 */

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

#include "kernel.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;
using namespace cv;

typedef vector<Mat> GPyramid;
typedef vector<Mat> LPyramid;

class Pyramid {
public:
	Pyramid(const Kernel& k) : K(k) {}
	Pyramid(const Kernel& k, const Mat& img);
	Pyramid(const Pyramid& p);  // copy ctor, used in "completion.cpp"
	Pyramid& operator=(const Pyramid& p);
	void compute_gaussian_pyramid(const vector<RECT>& roi_vec);
	void compute_laplace_pyramid();
	Mat get_real_image(const int dst_level);  // combine Gaussian and Laplace
	void save_images();

	~Pyramid() {
		K.~Kernel();
		GImages.clear();
		LImages.clear();
	}
	Kernel K;

private:
	Mat down_sample(const int cur_level, const RECT& roi);
	Mat up_sample(const int cur_level);

	int level;  // "level" should be no more than 9
	// images in Gaussian pyramid: "level",
	// images in Laplace pyramid: "level" - 1
	GPyramid GImages;  // Gaussian pyramid
	LPyramid LImages;  // Laplace pyramid
};

#endif
