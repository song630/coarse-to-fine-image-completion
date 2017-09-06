#include "completion.h"

using namespace std;
using namespace cv;

#define MIN2(a, b) (((a) < (b)) ? (a) : (b))
#define KERNEL_SIZE 5  // opencv uses 5 * 5 Gaussian kernel

void completion::initialize()
{
	Seg.draw_rect();
	Mat masked = Seg.get_masked();
	Pyramid temp(Pyr.K, masked, src);  // call ctor
	// ===== notice: in class Pyramid, "=" function does not assign Kernel "K"
	Pyr = temp;
	roi_vec.push_back(Seg.get_rect());
	RECT r;
	while (true)  // the hole can no longer be zoomed out
	{
		r = roi_vec.back();
		// x = 6, after removal of even cols, x = 2.
		r.first.x = r.first.x / 2;
		r.first.y = r.first.y / 2;
		r.second.x = r.second.x / 2 - (r.second.x % 2 == 0);
		r.second.y = r.second.y / 2 - (r.second.y % 2 == 0);
		cout << "r.first.x: " << r.first.x << ", r.first.y: " << r.first.y << ", ";
		cout << "r.second.x: " << r.second.x << ", r.second.y: " << r.second.y << endl;
		if (r.second.x - r.first.x + 1 < 2 || r.second.y - r.first.y + 1 < 2)
			break;  // the side length of the hole is less than 2
		roi_vec.push_back(r);
	}
	Pyr.compute_src_pyramid(roi_vec);
	Pyr.compute_gaussian_pyramid(roi_vec);
	Pyr.compute_laplace_pyramid(roi_vec);
	cout << "building pyramid: finished." << endl;
	Pyr.save_images();
}

string name1 = "rst1.jpg";
// string name2 = "real1.jpg";

Mat completion::image_complete()
{
	vector<vector<patch> > PATCHES;
	Mat rst;
	// traverse the pyramid reversely, from coarse to fine
	for (vector<RECT>::reverse_iterator i = roi_vec.rbegin(); i != roi_vec.rend(); i++)
	{
		int cur_level = distance(i, roi_vec.rend()) - 1;  // dist between two iterators
		cout << "current level: " << cur_level << endl;
		PatchMatch PM(Pyr.get_real_image(cur_level, *i), (*i), PATCHES);  // ctor
		PM.init();
		rst = PM.propagation_search();
		imwrite("D://from_ImageNet/" + name1, rst);
		name1[3]++;
		PATCHES = PM.cur_PATCHES;  // update. to be propagated in next iteration
	}
	return rst;
}
