/* segmentation.h
 * 1. the user draw a rectangular including the hole to be completed
 * 2. call grabCut() to get the object
 * 3. compute the mask
 */

#ifndef _SEGMENTATION_H_
#define _SEGMENTATION_H_

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <utility>

using namespace std;
using namespace cv;

// use top-left and bottom-right points to represent a rect
typedef pair<Point, Point> RECT;

class segmentation {
public:
	segmentation(const Mat& img) : src(img) {}
	~segmentation() {}
	void draw_rect();
	void print() {
		cout << rect_region.first.x << ", " << rect_region.first.y << endl;
	}
	Mat get_masked();
	RECT get_rect() {
		return rect_region;
	}

private:
	void on_mouse(int event, int x, int y, int flags);

	// without the function below, setMouseCallback() cannot call on_mouse.
	// from https://stackoverflow.com/questions/14280220/how-to-use-cvsetmousecallback/14281914#14281914
	static void on_mouse(int event, int x, int y, int flags, void *ustc) {
		static_cast<segmentation*>(ustc)->on_mouse(event, x, y, flags);
	}

	RECT rect_region;  // "roi" in patchmatch.h
	Mat src, disp;  // "disp": used for dispalying in on_mouse()
};

#endif
