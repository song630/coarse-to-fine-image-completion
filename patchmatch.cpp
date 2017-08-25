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
		cout << "Incorrect image size." << endl;
		exit(1);
	}
	// notice: in "Point", x-column, y-row.
	if (roi.first.x < SIDE_LEN - 1 || roi.first.y < SIDE_LEN - 1
		|| roi.second.x > width - SIDE_LEN || roi.second.y > height - SIDE_LEN)
	{
		cout << "roi.first.x: " << roi.first.x << ", ";
		cout << "roi.first.y: " << roi.first.y << ", ";
		cout << "roi.second.x: " << roi.second.x << ", ";
		cout << "roi.second.y: " << roi.second.y << endl;
		cout << "height: " << height << ", " << "width: " << width << endl;
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

// set "location" and "offset", and compute "similarity"
void PatchMatch::init()
{
	int query_col_start = roi.first.x - SIDE_LEN + 1;
	int query_col_end = roi.second.x + SIDE_LEN - 1;
	int query_row_start = roi.first.y - SIDE_LEN + 1;
	int query_row_end = roi.second.y + SIDE_LEN - 1;
	cur_PATCHES.resize(query_row_end - query_row_start + 1);

	if (pre_PATCHES.size() == 0)  // the top level, randomly assign colors
	{
		for (int i = query_row_start; i <= query_row_end; i++)  // resize "cur_PATCHES[]"
		{  // traverse every pixel 
			cur_PATCHES[i].resize(query_col_end - query_col_start + 1);
			for (int j = query_col_start; j <= query_col_end; j++)
			{
				patch temp(j, i);  // call ctor
				cur_PATCHES[i][j] = temp;

				srand((unsigned)time(NULL));
				// rand_x, rand_y: coordinates of the randomly assigned patch
				int rand_x = std::rand() % (width - SIDE_LEN + 1);  // 0 to width - 3 + 1
				int rand_y = std::rand() % (height - SIDE_LEN + 1);  // 0 to height - 3 + 1
				while (rand_x > roi.first.x - SIDE_LEN && rand_x < roi.second.x + SIDE_LEN  // within the hole
					&& rand_y > roi.first.y - SIDE_LEN && rand_y < roi.second.y + SIDE_LEN)
				{  // assign again
					// ===== srand() again ?
					rand_x = std::rand() % (width - SIDE_LEN + 1);
					rand_y = std::rand() % (height - SIDE_LEN + 1);
				}
				cur_PATCHES[i][j].update(Point(rand_x - j, rand_y - i), get_simil(q_patches[i][j], q_patches[rand_y][rand_x]));
			}
		}  // end-traverse
		// then compute the colors:
		color_update();
	}
	else  // the even rows and even cols need to be filled
	{
		for (int i = query_row_start, m = 1; i <= query_row_end; i++, m++)  // resize "cur_PATCHES[]"
		{  // traverse every pixel 
			cur_PATCHES[i].resize(query_col_end - query_col_start + 1);
			// m and n: counter
			for (int j = query_col_start, n = 1; j <= query_col_end; j++, n++)
			{
				if (m % 2 == 1 && n % 2 == 1)  // has a point in "pre_PATCHES[]"
					cur_PATCHES[i][j].propagate(pre_PATCHES[m][n], Point(i, j), 0, 0);
				else if (m % 2 == 1 && n % 2 == 0)  // refer to the point on the left
					cur_PATCHES[i][j].propagate(pre_PATCHES[m][n - 1], Point(i, j), -1, 0);
				else if (m % 2 == 0 && n % 2 == 1)  // refer to the point up
					cur_PATCHES[i][j].propagate(pre_PATCHES[m - 1][n], Point(i, j), 0, -1);
				else  // refer to the point on the top-left
					cur_PATCHES[i][j].propagate(pre_PATCHES[m - 1][n - 1], Point(i, j), -1, -1);
			}
		}
		// compute the colors:
		float r_color, g_color, b_color;
		for (int i = roi.first.y, a = 1; i <= roi.second.y; i++, a++)  // traverse the hole
		{
			Vec3b *p1 = query.ptr<Vec3b>(i);  // get the first pixel of row i
			Vec3b *p2 = query.ptr<Vec3b>(i - 1);  // row (i - 1)
			for (int j = roi.first.x, b = 1; j <= roi.second.x; j++, b++)
			{  // a and b: counters
				if (a % 2 == 1 && b % 2 == 0)  // refer to the point on the left
				{
					p1[j][0] = p1[j - 1][0];
					p1[j][1] = p1[j - 1][1];
					p1[j][2] = p1[j - 1][2];
					continue;
				}
				else if (a % 2 == 0 && b % 2 == 1)  // refer to the point up
				{
					p1[j][0] = p2[j][0];
					p1[j][1] = p2[j][1];
					p1[j][2] = p2[j][2];
					continue;
				}
				else if (a % 2 == 0 && b % 2 == 0)  // refer to the point on the top-left
				{
					p1[j][0] = p2[j - 1][0];
					p1[j][1] = p2[j - 1][1];
					p1[j][2] = p2[j - 1][2];
					continue;
				}
				r_color = g_color = b_color = 0;
				// ===== notice: some "dst_pixel" within the hole may be black ?
				for (int m = i - SIDE_LEN + 1; m <= i + SIDE_LEN - 1; m++)  // traverse all patches containing a pixel
				{
					for (int n = j - SIDE_LEN + 1; n <= j + SIDE_LEN - 1; n++)
					{
						Point dst_pixel;
						dst_pixel.x = cur_PATCHES[m][n].get_offset().x + j;
						dst_pixel.y = cur_PATCHES[m][n].get_offset().y + i;
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
	}  // end-else
}

float PatchMatch::get_simil(const Mat& a, const Mat& b)  // private
{
	Mat rst;  // an 1 * 4 Scaler
	absdiff(a, b, rst);  // compute the difference
	return static_cast<float>(sum(sum(rst))[0]);
}

Mat PatchMatch::propagation_search()
{
	float s1, s2;  // difference of patch1 and patch2
	int query_col_start = roi.first.x - SIDE_LEN + 1;
	int query_row_start = roi.first.y - SIDE_LEN + 1;
	for (int i = 0; i <= ITERATIONS - 1; i++)
	{
		if (i % 2 != 0)  // odd iterations
		{
			int r_x, r_y;  // coordinates of the relative patch
			PthOfImg::iterator iter1 = cur_PATCHES.begin();
			for (int j = query_row_start; iter1 != cur_PATCHES.end(); iter1++, j++)
			{  // traverse every patch
				vector<patch>::iterator iter2 = (*iter1).begin();
				for (int k = query_col_start; iter2 != (*iter1).end(); iter2++, k++)
				{
					r_x = (*iter2).get_offset().x + k;  // col
					r_y = (*iter2).get_offset().y + j;  // row
					s1 = get_simil(q_patches[j][k], q_patches[r_y][r_x - 1]);  // left
					s2 = get_simil(q_patches[j][k], q_patches[r_y - 1][r_x]);  // up
					// then compare and update the mapping
					float min_s = std::min((*iter2).get_simil(), MIN2(s1, s2));
					if (min_s == s1)  // left
						(*iter2).update(Point(r_x - 1 - k, r_y - j), s1);  // patch 1 pixel left
					else if (min_s == s2)  // up
						(*iter2).update(Point(r_x - k, r_y - 1 - j), s2);  // patch 1 pixel above
				}
			}
		}
		else  // even iterations, scan reversely
		{
			int r_x, r_y;  // coordinates of the relative patch
			for (int j = roi.second.y; j >= query_row_start; j--)  // scan from the last row
			{
				for (int k = roi.second.x; k >= query_col_start; k--)  // scan from the last column
				{
					r_x = cur_PATCHES[j][k].get_offset().x + k;  // col
					r_y = cur_PATCHES[j][k].get_offset().y + j;  // row
					s1 = get_simil(q_patches[j][k], q_patches[r_y + 1][r_x]);  // down
					s2 = get_simil(q_patches[j][k], q_patches[r_y][r_x + 1]);  // right
					// then compare and update the mapping
					float min_s = std::min(cur_PATCHES[j][k].get_simil(), MIN2(s1, s2));
					if (min_s == s1)
						cur_PATCHES[j][k].update(Point(r_x - k, r_y + 1 - j), s1);  // down
					else if (min_s == s2)
						cur_PATCHES[j][k].update(Point(r_x + 1 - k, r_y - j), s2);  // right
				}
			}
		}

		// random search:
		int rand_x, rand_y;  // coordinates of patch2
		float s;  // similarity
		PthOfImg::iterator iter1 = cur_PATCHES.begin();
		for (int j = query_row_start; iter1 != cur_PATCHES.end(); iter1++, j++)
		{
			vector<patch>::iterator iter2 = (*iter1).begin();
			for (int k = query_col_start; iter2 != (*iter1).end(); iter2++, k++)
			{
				int search_x = width - SIDE_LEN + 1;  // initial search radius
				int search_y = height - SIDE_LEN + 1;
				search_x >>= 1;
				search_y >>= 1;
				int left_bound, right_bound, up_bound, down_bound;
				while (search_x > 1 && search_y > 1)  // break when radius is less than 1 pixel
				{  // first compute the range:
					Point off = (*iter2).get_offset();
					left_bound = MAX2(0, k + off.x - search_x);
					right_bound = MIN2(width - SIDE_LEN, k + off.x + search_x);
					up_bound = MAX2(0, j + off.y - search_y);
					down_bound = MIN2(height - SIDE_LEN, j + off.y + search_y);
					srand((unsigned)time(NULL));  // then get a random patch:
					rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
					rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
					while ((rand_x - k) * (rand_x - k) + (rand_y - j) * (rand_y - j) >
						PatchMatch::threshold * PatchMatch::threshold)
					{
						rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
						rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
					}
					s = get_simil(q_patches[j][k], q_patches[rand_y][rand_x]);
					if (s < (*iter2).get_simil())  // at last update
						(*iter2).update(Point(rand_x - k, rand_y - j), s);
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
	float r_color, g_color, b_color;
	for (int i = roi.first.y; i <= roi.second.y; i++)  // traverse the hole
	{
		Vec3b *p1 = query.ptr<Vec3b>(i);  // get the first pixel of row i
		for (int j = roi.first.x; j <= roi.second.x; j++)
		{
			r_color = g_color = b_color = 0;
			for (int m = i - SIDE_LEN + 1; m <= i + SIDE_LEN - 1; m++)  // traverse all patches containing a pixel
			{
				for (int n = j - SIDE_LEN + 1; n <= j + SIDE_LEN - 1; n++)
				{  // the location of "dst_pixel" in the relative patch,
					// is just the location of origin pixel in patch being visited.
					Point dst_pixel;
					dst_pixel.x = cur_PATCHES[m][n].get_offset().x + j;
					dst_pixel.y = cur_PATCHES[m][n].get_offset().y + i;
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
