#include "patch.h"
#include "patchmatch.h"
#include <ctime>

#define IMG_PATH "D://from_ImageNet/"  // the file where the images are saved
#define ITERATIONS 6
#define MAX_SIMIARITY 1e6
#define MAX2(a, b) (((a) > b) ? (a) : (b))
#define MIN2(a, b) (((a) < b) ? (a) : (b))

using namespace std;
using namespace cv;

PatchMatch::PatchMatch(const string& image1, const string& image2)  // ctor
{
	Mat temp1, temp2;
	temp1 = imread(IMG_PATH + image1);  // read the first image
	temp2 = imread(IMG_PATH + image2);  // read the second image
	// if image1 and image2 are of different size:
	width = MIN2(temp1.cols, temp2.cols);
	height = MIN2(temp1.rows, temp2.rows);
	cout << "width = " << width << ", " << "height = " << height << endl;
	query = temp1(Rect(0, 0, width, height));
	candidate = temp2(Rect(0, 0, width, height));
	if (width <= SIDE_LEN || height <= SIDE_LEN)
	{  // the image is smaller than a patch
		cout << "Incorrect image size." << endl;
		exit(-1);
	}
}

// notice: class "PatchMatch" does not have a member named "mode"
// set "location" and "offset", and compute "similarity"
void PatchMatch::init(const INIT_METHOD mode = direct)
{
	// first resize PthOfImg
	PATCHES.resize(height - SIDE_LEN + 1);  // from row 0 to ...
	q_patches.resize(height - SIDE_LEN + 1);
	c_patches.resize(height - SIDE_LEN + 1);
	for (int k = 0; k <= height - SIDE_LEN; k++)
	{
		PATCHES[k].resize(width - SIDE_LEN + 1);  // from column 0 to ...
		q_patches[k].resize(width - SIDE_LEN + 1);
		c_patches[k].resize(width - SIDE_LEN + 1);
		for (int m = 0; m <= width - SIDE_LEN; m++)
		{
			q_patches[k][m] = query(Rect(m, k, SIDE_LEN, SIDE_LEN));
			c_patches[k][m] = candidate(Rect(m, k, SIDE_LEN, SIDE_LEN));
		}
	}

	Mat patch1, patch2;
	int i = 0, j = 0;
	PthOfImg::iterator iter1;
	if (mode == direct)  // directly assign offset = 0
	{
		for (i = 0, iter1 = PATCHES.begin(); iter1 != PATCHES.end(); iter1++, i++)
		{
			vector<patch>::iterator iter2;
			for (j = 0, iter2 = (*iter1).begin(); iter2 != (*iter1).end(); iter2++, j++)
			{
				patch temp(i, j);
				(*iter2) = temp;  // get "location" for *iter2
				// compute "similarity": (first get the two patches)
				// i: row, j: column
				(*iter2).update(Point(0, 0), get_simil(q_patches[i][j], c_patches[i][j]));
			}
		}
	}
	else  // mode == random
	{
		int rand_x, rand_y;  // coordinates of the randomly generated patch
		for (i = 0, iter1 = PATCHES.begin(); iter1 != PATCHES.end(); iter1++, i++)
		{
			vector<patch>::iterator iter2;
			for (j = 0, iter2 = (*iter1).begin(); iter2 != (*iter1).end(); iter2++, j++)
			{
				patch temp(i, j);
				(*iter2) = temp;  // get "location" for *iter2
				// compute "similarity" (first get the two patches):
				// first get random coordinates:
				srand((unsigned)time(NULL));
				rand_x = std::rand() % (height - SIDE_LEN + 1);  // 0 to height - 5 + 1
				rand_y = std::rand() % (width - SIDE_LEN + 1);  // 0 to width - 5 + 1
				(*iter2).update(Point(rand_x - i, rand_y - j), get_simil(q_patches[i][j], c_patches[rand_x][rand_y]));
			}
		}
	}  // end of else-block
}

float PatchMatch::get_simil(const Mat& a, const Mat& b)  // private
{
	Mat rst;  // an 1 * 4 Scaler
	absdiff(a, b, rst);  // compute the difference
	return static_cast<float>(sum(sum(rst))[0]);
}

void PatchMatch::propagation_search(const INIT_METHOD mode = direct)
{
	float s1, s2;  // difference of patch1 and patch2

	for (int i = 0; i <= ITERATIONS - 1; i++)
	{  // value of "step" differs in even and odd iterations
		if (i % 2 != 0)  // odd iterations
		{
			int j = 0, k = 0;
			int r_x, r_y;  // coordinates of the relative patch
			PthOfImg::iterator iter1;
			// traverse every patch
			for (iter1 = PATCHES.begin(); iter1 != PATCHES.end(); iter1++, j++)
			{
				vector<patch>::iterator iter2;
				k = 0;
				// if (j % 10 == 0)
				// cout << "propagation: in row " << j << ", " << "iteration " << i << endl;
				for (iter2 = (*iter1).begin(); iter2 != (*iter1).end(); iter2++, k++)
				{
					if (mode == random)  // randomly assigned in init()
					{
						r_x = (*iter2).get_offset().x + j;
						r_y = (*iter2).get_offset().y + k;
					}
					else  // mode == direct
					{
						r_x = j;
						r_y = k;
					}

					if (r_x == 0 && r_y == 0)  // the patch on the top-left
						continue;
					else if (r_y == 0)  // j != 0. patches on the left column
					{
						s1 = get_simil(q_patches[j][k], c_patches[r_x - 1][r_y]);
						s2 = MAX_SIMIARITY;
					}
					else if (r_x == 0)  // k != 0. patches on the up row
					{
						s2 = get_simil(q_patches[j][k], c_patches[r_x][r_y - 1]);
						s1 = MAX_SIMIARITY;
					}
					else  // j != 0 && k != 0
					{
						s1 = get_simil(q_patches[j][k], c_patches[r_x - 1][r_y]);
						s2 = get_simil(q_patches[j][k], c_patches[r_x][r_y - 1]);
					}

					// then compare and update the mapping
					float min_s = std::min((*iter2).get_simil(), MIN2(s1, s2));
					if (min_s == s1)  // up
						(*iter2).update(Point(r_x - 1 - j, r_y - k), s1);  // patch 1 pixel above
					else if (min_s == s2)  // left
						(*iter2).update(Point(r_x - j, r_y - 1 - k), s2);  // patch 1 pixel left
				}
			}
		}
		else  // even iteratoins, scan reversely
		{
			int j = height - SIDE_LEN;
			int k = width - SIDE_LEN;
			int r_x, r_y;  // coordinates of the relative patch
			for (; j >= 0; j--)  // scan from the last row
			{
				k = width - SIDE_LEN;
				// if (j % 10 == 0)
				// cout << "propagation: in row " << j << ", " << "iteration " << i << endl;
				for (; k >= 0; k--)  // scan from the last column
				{
					if (mode == random)
					{  // find the relative patch in "candidate" image
						r_x = PATCHES[j][k].get_offset().x + j;
						r_y = PATCHES[j][k].get_offset().y + k;
					}
					else  // mode == direct
					{
						r_x = j;
						r_y = k;
					}
					if (r_x == height - SIDE_LEN && r_y == width - SIDE_LEN)
						continue;
					else if (r_x == height - SIDE_LEN)  // the lowest row
					{  // get the patch 1 pixel right
						s1 = get_simil(q_patches[j][k], c_patches[r_x][r_y + 1]);
						s2 = MAX_SIMIARITY;
					}
					else if (r_y == width - SIDE_LEN)  // the right-most column
					{  // get the patch 1 pixel up
						s2 = get_simil(q_patches[j][k], c_patches[r_x + 1][r_y]);
						s1 = MAX_SIMIARITY;
					}
					else
					{
						s1 = get_simil(q_patches[j][k], c_patches[r_x][r_y + 1]);
						s2 = get_simil(q_patches[j][k], c_patches[r_x + 1][r_y]);
					}

					// then compare and update the mapping
					float min_s = std::min(PATCHES[j][k].get_simil(), MIN2(s1, s2));
					if (min_s == s1)
						PATCHES[j][k].update(Point(r_x + 1 - j, r_y - k), s1);  // the patch above
					else if (min_s == s2)  // get the patch on the right
						PATCHES[j][k].update(Point(r_x - j, r_y + 1 - k), s2);
				}
			}
		}

		// random search:
		int j = 0, k = 0;  // coordinates of the patch in "query"
		int rand_x, rand_y;  // coordinates of patch2
		float s;  // similarity
		PthOfImg::iterator iter1;
		// traverse every patch
		for (iter1 = PATCHES.begin(); iter1 != PATCHES.end(); iter1++, j++)
		{
			k = 0;
			vector<patch>::iterator iter2;
			// if (j % 10 == 0)
			// cout << "random search: in row " << j << ", " << "iteration " << i << endl;
			for (iter2 = (*iter1).begin(); iter2 != (*iter1).end(); iter2++, k++)
			{
				int search_x = height - SIDE_LEN + 1;  // initial search radius
				int search_y = width - SIDE_LEN + 1;
				search_x >>= 1;
				search_y >>= 1;
				int left_bound, right_bound, up_bound, down_bound;
				while (search_x > 1 && search_y > 1)  // break when radius is less than 1 pixel
				{  // first compute the range:
					Point off = (*iter2).get_offset();
					left_bound = MAX2(0, k + off.y - search_y);
					right_bound = MIN2(width - SIDE_LEN, k + off.y + search_y);
					up_bound = MAX2(0, j + off.x - search_x);
					down_bound = MIN2(height - SIDE_LEN, j + off.x + search_x);
					srand((unsigned)time(NULL));  // then get a random patch:
					rand_x = std::rand() % (down_bound - up_bound + 1) + up_bound;
					rand_y = std::rand() % (right_bound - left_bound + 1) + left_bound;
					s = get_simil(q_patches[j][k], c_patches[rand_x][rand_y]);
					if (s < (*iter2).get_simil())  // at last update
						(*iter2).update(Point(rand_x - j, rand_y - k), s);
					search_x >>= 1;
					search_y >>= 1;
				}
			}
		}
		cout << "iteration " << i << endl;
	}  // end ITERATIONS
}

Mat PatchMatch::reshuffle()
{
	Mat recon;  // reconstructed image
	query.copyTo(recon);
	int i, j;
	Point off;  // get "offset"

	PthOfImg::iterator iter1;
	for (i = 0, iter1 = PATCHES.begin(); iter1 != PATCHES.end(); iter1++, i++)
	{
		vector<patch>::iterator iter2;
		for (j = 0, iter2 = (*iter1).begin(); iter2 != (*iter1).end(); iter2++, j++)
		{
			off = (*iter2).get_offset();
			recon.at<Vec3b>(i, j) = candidate.at<Vec3b>(i + off.x, j + off.y);
		}
	}
	// deal with rows at the bottom:

	// deal with columns at the right side:

	return recon;
}
