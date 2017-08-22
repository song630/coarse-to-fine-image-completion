#include "segmentation.h"

using namespace std;
using namespace cv;

void segmentation::get_rect()
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
		rect_region.first = Point(x, y);  // top-left point
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
		rect_region.second = Point(x, y);  // bottom-right point
		rectangle(disp, pre_pt, cur_pt, Scalar(255, 0, 0, 0), 1, 8, 0);  // display rectangular on temp Mat
		imshow("img", disp);
		waitKey(0);  // close window
	}
}

Mat segmentation::get_masked()
{
	// get the region where the object will be segmented
	Rect rectangle(rect_region.first.x, rect_region.first.y,
		rect_region.second.x - rect_region.first.x, rect_region.second.y - rect_region.first.y);
	Mat bgModel, fgModel;  // intermediate variables
	grabCut(src, mask, rectangle, bgModel, fgModel, 1, GC_INIT_WITH_RECT);  // get the object
	cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
	Mat masked;
	src.copyTo(masked);
	masked.setTo(0, mask);  // remove the object from the src image, generating a hole
	return masked;
}
