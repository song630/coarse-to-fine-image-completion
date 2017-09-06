#include "segmentation.h"

using namespace std;
using namespace cv;

void segmentation::draw_rect()
{
	namedWindow("img");
	setMouseCallback("img", on_mouse, this);  // call on_mouse()
	imshow("img", src);
	waitKey(0);
}

void segmentation::on_mouse(int event, int x, int y, int flags)
{
	static Point pre_pt = (-1, -1);  // init coordinates
	static Point cur_pt = (-1, -1);  // current coordinates
	if (event == CV_EVENT_LBUTTONDOWN)  // left-button down: get init coordinates 
	{
		pre_pt = Point(x, y);
		rect_region.first = Point(x - 2, y - 2);  // top-left point
	}
	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))  // left-button up, mouse moving
		cur_pt = Point(x, y);
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))  // left-button down, mouse moving  
	{
		src.copyTo(disp);
		cur_pt = Point(x, y);
		rectangle(disp, pre_pt, cur_pt, Scalar(255, 0, 0, 0), 1, 8, 0);  // display rectangular on temp Mat 
		imshow("img", disp);
	}
	else if (event == CV_EVENT_LBUTTONUP)  // left-botton up
	{
		src.copyTo(disp);
		cur_pt = Point(x, y);
		rect_region.second = Point(x + 2, y + 2);  // bottom-right point
		Point temp = rect_region.first;  // to be copied
		if (rect_region.first.x > rect_region.second.x)
		{  // exchange x coordinate
			rect_region.first.x = rect_region.second.x;
			rect_region.second.x = temp.x;
		}
		if (rect_region.first.y > rect_region.second.y)
		{  // exchange y coordinate
			rect_region.first.y = rect_region.second.y;
			rect_region.second.y = temp.y;
		}
		rectangle(disp, pre_pt, cur_pt, Scalar(255, 0, 0, 0), 1, 8, 0);  // display rectangular on temp Mat
		imshow("img", disp);
		waitKey(0);  // close window
	}
}

Mat segmentation::get_masked()
{
	// get the region where the object will be segmented
	// ===== notice: the pixel "rect_region.first" will be black, but "rect_region.second" will not
	// ===== correction: "+ 1" added
	Rect rectangle(rect_region.first.x, rect_region.first.y,
		rect_region.second.x - rect_region.first.x + 1, rect_region.second.y - rect_region.first.y + 1);
	Mat mask, masked;
	mask = Mat::zeros(src.size(), CV_8UC1);  // build an image with all pixels initialzed to 0
	mask(rectangle).setTo(255);  // set the hole white
	src.copyTo(masked);
	masked.setTo(0, mask);  // remove the object from the src image, generating a hole
	return masked;
}
