/* pyramid.h
 * both Gaussian and Laplace pyramid included.
 */

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

#include "kernel.h"

using namespace std;
using namespace cv;

class Pyramid {
public:
	Pyramid(const Kernel& K, Mat& src);

	void get_gaussian_pyramid();
	void get_laplace_pyramid();
	Mat get_real_image(const int dst_level);  // combine Gaussian and Laplace
	void save_images();

	~Pyramid() {
		kernel.~Kernel();
		GImages.clear();
		LImages.clear();
	}

private:
	int level;  // "level" should be no more than 9
	Kernel kernel;
	typedef vector<Mat> GPyramid;
	typedef vector<Mat> LPyramid;
	// images in Gaussian pyramid: "level",
	// images in Laplace pyramid: "level" - 1
	GPyramid GImages;  // Gaussian pyramid
	LPyramid LImages;  // Laplace pyramid

	Mat down_sample(const int cur_level);  // get the image on level i + 1
	Mat up_sample(const int cur_level);  // get the image on level i - 1
};

#endif
