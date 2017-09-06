#include "patch.h"
#include "patchmatch.h"
#include <ctime>

#define IMG_PATH "D://from_ImageNet/"  // the file where the images are saved
#define ITERATIONS 5
#define MAX_SIMIARITY 1e6
#define MAX2(a, b) (((a) > (b)) ? (a) : (b))
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

float PatchMatch::alpha = 0.5;
int PatchMatch::threshold = 5;

// ===== threshold ?

PatchMatch::PatchMatch(const Mat& img, const RECT& _roi, const PthOfImg& pre_patches) :
query(img), roi(_roi), pre_PATCHES(pre_patches)  // ctor
{
	width = query.cols;
	height = query.rows;
	if (width <= SIDE_LEN || height <= SIDE_LEN)
	{  // the image is smaller than a patch
		cout << "Incorrect image size: " << endl;
		cout << "width: " << width << ", height: " << height << endl;
		system("pause");
		exit(1);
	}
	// notice: in "Point", x-column, y-row.
	if (roi.first.x == 0 || roi.first.y == 0 ||
		roi.second.x == width - 1 || roi.second.y == height - 1)
	{
		cout << "The hole is near the boundary." << endl;
		system("pause");
		exit(1);
	}

	q_patches.resize(height - SIDE_LEN + 1);  // resize "q_patches"
	for (int i = 0; i <= height - SIDE_LEN; i++)
	{
		q_patches[i].resize(width - SIDE_LEN + 1);
		for (int j = 0; j <= width - SIDE_LEN; j++)
			q_patches[i][j] = query(Rect(j, i, SIDE_LEN, SIDE_LEN));  // get all patches in the image
	}
}

PatchMatch::~PatchMatch()
{
	PthOfImg::iterator i1;
	for (i1 = cur_PATCHES.begin(); i1 != cur_PATCHES.end(); i1++)
		(*i1).clear();
	cur_PATCHES.clear();

	for (i1 = pre_PATCHES.begin(); i1 != pre_PATCHES.end(); i1++)
		(*i1).clear();
	pre_PATCHES.clear();

	PatchInQ::iterator i3;
	for (i3 = q_patches.begin(); i3 != q_patches.end(); i3++)
		(*i3).clear();
	q_patches.clear();
}

string name = "init1.jpg";
string name3 = "before_init1.jpg";

// set "location" and "offset", and compute "similarity"
void PatchMatch::init()
{
	int q_col_start = roi.first.x - SIDE_LEN + 1;
	int q_row_start = roi.first.y - SIDE_LEN + 1;
	cur_PATCHES.resize(roi.second.y - q_row_start + 1);

	// ===== DIFFERENT FROM "q_patches[]", IN "cur_PATCHES[][]",
	// SUBSCRIPT DOES NOT EQUAL TO ITS LOCATION.

	if (pre_PATCHES.size() == 0)  // the next-to-top level, randomly assign colors
	{
		for (int i = q_row_start; i <= roi.second.y; i++)  // resize "cur_PATCHES[]"
		{  // traverse every pixel
			cur_PATCHES[i - q_row_start].resize(roi.second.x - q_col_start + 1);
			for (int j = q_col_start; j <= roi.second.x; j++)
			{
				patch temp(j, i);  // call ctor
				cur_PATCHES[i - q_row_start][j - q_col_start] = temp;

				// rand_x, rand_y: coordinates of the randomly assigned patch
				int rand_x = std::rand() % (width - SIDE_LEN + 1);  // 0 to width - 3 + 1
				int rand_y = std::rand() % (height - SIDE_LEN + 1);  // 0 to height - 3 + 1
				while (rand_x > roi.first.x - SIDE_LEN && rand_x <= roi.second.x  // within the hole
					&& rand_y > roi.first.y - SIDE_LEN && rand_y <= roi.second.y)
				{  // assign again
					rand_x = std::rand() % (width - SIDE_LEN + 1);
					rand_y = std::rand() % (height - SIDE_LEN + 1);
				}
				cur_PATCHES[i - q_row_start][j - q_col_start].update_offset(Point(rand_x - j, rand_y - i));
			}
		}  // end-traverse
		// then compute the colors and update similarity:
		color_update();
		for (int i = q_row_start; i <= roi.second.y; i++)
		{
			for (int j = q_col_start; j <= roi.second.x; j++)
			{
				patch temp_p = cur_PATCHES[i - q_row_start][j - q_col_start];
				int dst_patch_x = temp_p.get_offset().x + j;
				int dst_patch_y = temp_p.get_offset().y + i;
				temp_p.update_sim(get_sim(q_patches[i][j], q_patches[dst_patch_y][dst_patch_x]));
			}
		}
		imwrite("D://from_ImageNet/" + name, query);
		name[4]++;
	}
	else  // the even rows and even cols need to be filled
	{
		imwrite("D://from_ImageNet/" + name3, query);
		name3[11]++;

		Point pre_region(roi.first.x / 2 - SIDE_LEN + 1, roi.first.y / 2 - SIDE_LEN + 1);
		cur_PATCHES.resize(roi.second.y - q_row_start + 1);
		for (int i = q_row_start; i <= roi.second.y; i++)
		{
			cur_PATCHES[i - q_row_start].resize(roi.second.x - q_col_start + 1);
			for (int j = q_col_start; j <= roi.second.x; j++)
			{  // find the most similar patch at this level:
				// "dst_patch": the relative patch of cur_PATCHES[][] at upper level
				patch dst_patch = pre_PATCHES[i / 2 - 1 - pre_region.y][j / 2 - 1 - pre_region.x];
				Point most_sim;  // coordinates of the most similar patch of cur_PATCHES[][] at current level
				most_sim.x = (j / 2 - 1 + dst_patch.get_offset().x) * 2;
				most_sim.y = (i / 2 - 1 + dst_patch.get_offset().y) * 2;
				if (most_sim.x > width - SIDE_LEN)
					most_sim.x--;
				if (most_sim.y > height - SIDE_LEN)
					most_sim.y--;
				cur_PATCHES[i - q_row_start][j - q_col_start].update_offset(Point(most_sim.x - j, most_sim.y - i));
				// compute similarity later (after color)
			}
		}
		// compute the colors:
		color_update();
		// compute similarity:
		for (int i = q_row_start; i <= roi.second.y; i++)
		{
			for (int j = q_col_start; j <= roi.second.x; j++)
			{
				int dst_x = j + cur_PATCHES[i - q_row_start][j - q_col_start].get_offset().x;
				int dst_y = i + cur_PATCHES[i - q_row_start][j - q_col_start].get_offset().y;
				cur_PATCHES[i - q_row_start][j - q_col_start].update_sim(get_sim(q_patches[i][j], q_patches[dst_y][dst_x]));
			}
		}
		imwrite("D://from_ImageNet/" + name, query);
		name[4]++;
	}  // end-else
}

float PatchMatch::get_sim(const Mat& a, const Mat& b)  // private
{
	Mat rst;  // an 1 * 4 Scaler
	absdiff(a, b, rst);  // compute the difference
	return static_cast<float>(sum(sum(rst))[0]);
}

Mat PatchMatch::propagation_search()
{
	float s1, s2;  // difference of patch1 and patch2
	int q_col_start = roi.first.x - SIDE_LEN + 1;
	int q_row_start = roi.first.y - SIDE_LEN + 1;
	for (int i = 1; i <= ITERATIONS; i++)
	{
		cout << "begin iteration " << i << endl;
		if (i % 2 != 0)  // odd iterations
		{
			int r_x, r_y;  // coordinates of the relative patch
			PthOfImg::iterator iter1 = cur_PATCHES.begin();
			for (int j = q_row_start; iter1 != cur_PATCHES.end(); iter1++, j++)
			{  // traverse every patch
				vector<patch>::iterator iter2 = (*iter1).begin();
				for (int k = q_col_start; iter2 != (*iter1).end(); iter2++, k++)
				{
					r_x = (*iter2).get_offset().x + k;  // col
					r_y = (*iter2).get_offset().y + j;  // row
					// in case "r_y - 1" or "r_x - 1" is out of range:
					s1 = (r_x == 0) ? MAX_SIMIARITY : get_sim(q_patches[j][k], q_patches[r_y][r_x - 1]);  // left
					s2 = (r_y == 0) ? MAX_SIMIARITY : get_sim(q_patches[j][k], q_patches[r_y - 1][r_x]);  // up
					// then compare and update the mapping
					float min_s = std::min((*iter2).get_sim(), MIN2(s1, s2));
					if (min_s == s1)  // left
					{  // patch 1 pixel left
						(*iter2).update_offset(Point(r_x - 1 - k, r_y - j));
						(*iter2).update_sim(s1);
					}
					else if (min_s == s2)  // up
					{
						(*iter2).update_offset(Point(r_x - k, r_y - 1 - j));
						(*iter2).update_sim(s2);
					}
				}
			}
		}
		else  // even iterations, scan reversely
		{
			int r_x, r_y;  // coordinates of the relative patch
			for (int j = roi.second.y; j >= q_row_start; j--)  // scan from the last row
			{
				for (int k = roi.second.x; k >= q_col_start; k--)  // scan from the last column
				{
					r_x = cur_PATCHES[j - q_row_start][k - q_col_start].get_offset().x + k;  // col
					r_y = cur_PATCHES[j - q_row_start][k - q_col_start].get_offset().y + j;  // row
					// in case "r_y + 1" or "r_x + 1" is out of range:
					s1 = (r_y >= query.rows - SIDE_LEN) ? MAX_SIMIARITY : get_sim(q_patches[j][k], q_patches[r_y + 1][r_x]);  // down
					s2 = (r_x >= query.cols - SIDE_LEN) ? MAX_SIMIARITY : get_sim(q_patches[j][k], q_patches[r_y][r_x + 1]);  // right
					// then compare and update the mapping
					float min_s = std::min(cur_PATCHES[j - q_row_start][k - q_col_start].get_sim(), MIN2(s1, s2));
					if (min_s == s1)
					{  // down
						cur_PATCHES[j - q_row_start][k - q_col_start].update_offset(Point(r_x - k, r_y + 1 - j));
						cur_PATCHES[j - q_row_start][k - q_col_start].update_sim(s1);
					}
					else if (min_s == s2)
					{  // right
						cur_PATCHES[j - q_row_start][k - q_col_start].update_offset(Point(r_x + 1 - k, r_y - j));
						cur_PATCHES[j - q_row_start][k - q_col_start].update_sim(s2);
					}
				}
			}
		}
		cout << "random search in iteration " << i << endl;
		// random search:
		int rand_x, rand_y;  // coordinates of patch2
		float s;  // similarity
		PthOfImg::iterator iter1 = cur_PATCHES.begin();
		for (int j = q_row_start; iter1 != cur_PATCHES.end(); iter1++, j++)
		{
			vector<patch>::iterator iter2 = (*iter1).begin();
			for (int k = q_col_start; iter2 != (*iter1).end(); iter2++, k++)
			{
				int search_x = width - SIDE_LEN + 1;  // initial search radius
				int search_y = height - SIDE_LEN + 1;
				search_x >>= 1;
				search_y >>= 1;
				int left_bound, right_bound, up_bound, down_bound;
				while (search_x > 3 && search_y > 3)  // break when radius is less than 1 pixel
				{  // first compute the range:
					Point off = (*iter2).get_offset();
					left_bound = MAX2(0, k + off.x - search_x);
					right_bound = MIN2(width - SIDE_LEN, k + off.x + search_x);
					up_bound = MAX2(0, j + off.y - search_y);
					down_bound = MIN2(height - SIDE_LEN, j + off.y + search_y);
					srand((unsigned)time(NULL));  // then get a random patch:
					rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
					rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
					// cout << "j: " << j << ", " << "k: " << k << endl;
					while ((rand_x - k) * (rand_x - k) + (rand_y - j) * (rand_y - j) <
						PatchMatch::threshold * PatchMatch::threshold)
					{
						rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
						rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
						// cout << "rand_y: " << rand_y << ", " << "rand_x: " << rand_x << endl;
					}
					s = get_sim(q_patches[j][k], q_patches[rand_y][rand_x]);
					if (s < (*iter2).get_sim())  // at last update
					{
						(*iter2).update_offset(Point(rand_x - k, rand_y - j));
						(*iter2).update_sim(s);
					}
					search_x >>= 1;
					search_y >>= 1;
				}
			}
		}
		color_update();
		cout << "iteration " << i << endl;
	}  // end ITERATIONS
	return query;
}

void PatchMatch::color_update()
{
	int q_col_start = roi.first.x - SIDE_LEN + 1;
	int q_row_start = roi.first.y - SIDE_LEN + 1;
	float r_color, g_color, b_color;
	cout << "in color_update(): roi.first.x = " << roi.first.x << ", roi.first.y = " << roi.first.y << endl;
	for (int i = roi.first.y; i <= roi.second.y; i++)  // traverse the hole
	{
		Vec3b *p1 = query.ptr<Vec3b>(i);  // get the first pixel of row i
		for (int j = roi.first.x; j <= roi.second.x; j++)
		{
			r_color = g_color = b_color = 0;
			for (int m = i - SIDE_LEN + 1; m <= i; m++)  // traverse all patches containing a pixel
			{
				for (int n = j - SIDE_LEN + 1; n <= j; n++)
				{   // the location of "dst_pixel" in the relative patch,
					// which is just the location of original pixel in patch being visited.
					Point dst_pixel;
					dst_pixel.x = cur_PATCHES[m - q_row_start][n - q_col_start].get_offset().x + j;
					dst_pixel.y = cur_PATCHES[m - q_row_start][n - q_col_start].get_offset().y + i;
					// cout << "dst_pixel.x: " << dst_pixel.x << ", dst_pixel.y: " << dst_pixel.y << endl;
					// ===== unfixed bug here
					b_color += query.at<Vec3b>(dst_pixel)[0];
					g_color += query.at<Vec3b>(dst_pixel)[1];
					r_color += query.at<Vec3b>(dst_pixel)[2];
				}
			}
			p1[j][0] = static_cast<unsigned char>(b_color / (SIDE_LEN * SIDE_LEN));  // blue
			p1[j][1] = static_cast<unsigned char>(g_color / (SIDE_LEN * SIDE_LEN));  // green
			p1[j][2] = static_cast<unsigned char>(r_color / (SIDE_LEN * SIDE_LEN));  // red
		}
	}
}
