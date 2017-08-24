/* pyramid.h
 * both Gaussian and Laplace pyramid included.
 */

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

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
	Pyramid(const Mat& img);
	void compute_gaussian_pyramid();
	void compute_laplace_pyramid();
	Mat get_real_image(const int dst_level);  // combine Gaussian and Laplace
	void save_images();

	~Pyramid() {
		// kernel.~Kernel();
		GImages.clear();
		LImages.clear();
	}

private:
	int level;  // "level" should be no more than 9
	// images in Gaussian pyramid: "level",
	// images in Laplace pyramid: "level" - 1
	GPyramid GImages;  // Gaussian pyramid
	LPyramid LImages;  // Laplace pyramid
};

#endif
