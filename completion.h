/* completion.h
 * class completion includes class segmentation and class pyramid.
 */

#ifndef _COMPLETION_H_
#define _COMPLETION_H_

#include "segmentation.h"
#include "pyramid.h"
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class completion {  // image completion
public:
	completion(const Mat& img) : src(img), Seg(img), Pyr(img) {}
	~completion() {
		Pyr.~Pyramid();
		Seg.~segmentation();
		delete[] roi_ptr;
	}
	// init: fill the hole, build up a pyramid, randomly fill the image at the top level
	void initialize();

private:
	Mat src;
	segmentation Seg;  // fill the rectangular hole with black pixels
	Pyramid Pyr;  // build pyramid from the image with a hole
	static int iterations;  // to be set in main.cpp
	static int threshold;  // the most coherent patch cannot be itself
	RECT *roi_ptr;  // top-left and bottom-right points of every level
};

#endif
