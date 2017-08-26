#include "completion.h"

using namespace std;
using namespace cv;

#define MIN2(a, b) (((a) < (b)) ? (a) : (b))
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel

void completion::initialize()
{
	Seg.draw_rect();
	Mat masked = Seg.get_masked();
	Pyramid temp(masked);  // call ctor
	Pyr = temp;
	roi_vec.push_back(Seg.get_rect());
	int length = MIN2(masked.rows, masked.cols);
	int row = masked.rows;
	int col = masked.cols;
	RECT r;
	while (true)
	{  // get the range of the hole of every level
		r = roi_vec.back();
		// x = 6, after removal of even cols, x = 2.
		r.first.x = r.first.x / 2 - (r.first.x % 2 == 0);
		r.first.y = r.first.y / 2 - (r.first.y % 2 == 0);
		r.second.x = r.second.x / 2 + (r.second.x % 2 == 0);
		r.second.y = r.second.y / 2 + (r.second.y % 2 == 0);
		// cout << "r.first.x: " << r.first.x << ", ";
		// cout << "r.first.y: " << r.first.y << ", ";
		// cout << "r.second.x: " << r.second.x << ", ";
		// cout << "r.second.y: " << r.second.y << endl;
		length >>= 1;
		row >>= 1;
		col >>= 1;
		if (length <= 2 * KERNEL_SIZE || r.first.x < SIDE_LEN - 1 ||
			r.first.y < SIDE_LEN - 1 || r.second.x > col - SIDE_LEN
			|| r.second.y > row - SIDE_LEN)
			break;
		roi_vec.push_back(r);
	}
	Pyr.compute_gaussian_pyramid();
	Pyr.compute_laplace_pyramid();
	cout << "building pyramid: finished." << endl;
	Pyr.save_images();
}

Mat completion::image_complete()
{
	vector<vector<patch> > PATCHES;
	Mat rst;
	// traverse the pyramid reversely, from coarse to fine
	for (vector<RECT>::reverse_iterator i = roi_vec.rbegin(); i != roi_vec.rend(); i++)
	{
		int cur_level = distance(i, roi_vec.rend()) - 1;  // dist between two iterators
		cout << "current level: " << cur_level << endl;
		if (i == roi_vec.rbegin())
			imwrite("D://from_ImageNet/top.jpg", Pyr.get_real_image(cur_level));
		PatchMatch PM(Pyr.get_real_image(cur_level), (*i), PATCHES);  // ctor
		PM.init();
		cout << "initialize patchmatch: finished." << endl;
		rst = PM.propagation_search();
		PATCHES = PM.cur_PATCHES;  // update. to be propagated in next iteration
	}
	return rst;
}
