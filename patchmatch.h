#ifndef _PATCHMATCH_H_
#define _PATCHMATCH_H_

#pragma once

#include "patch.h"
#include <vector>
#include <string>

#define SIDE_LEN 3

using namespace std;
using namespace cv;

/*  two methods to init(), deciding initial offset for all patches:
*  rand: randomly distribute.
*  direct: initial offset = 0.
*/
enum INIT_METHOD { random, direct };

class PatchMatch {
public:
	PatchMatch(const string& image1, const string& image2);
	~PatchMatch() {
		PthOfImg::iterator iter = PATCHES.begin();
		for (; iter != PATCHES.end(); iter++)
			(*iter).clear();
		PATCHES.clear();
	}
	void init(const INIT_METHOD mode);
	void propagation_search(const INIT_METHOD mode);
	Mat reshuffle();  // reconstruction
	void print() {
		for (int i = 0; i <= height - SIDE_LEN; i += 10)
		{
			for (int j = 0; j <= width - SIDE_LEN; j += 10)
			{
				cout << "[" << PATCHES[i][j].get_offset().x << ", ";
				cout << PATCHES[i][j].get_offset().y << "]" << endl;
			}
		}
	}

private:
	Mat query, candidate;  // two source images
	int width, height;  // the two images must be of the same size

	// used in random search
	static float alpha;  // decay ratio

	static int step;  // i.e. every time move 1 pixel left or up
	// save all patches in an image, accessed by PthOfImg[][],
	// the size needs to be resized later (by width and height).
	typedef vector<vector<patch> > PthOfImg;
	PthOfImg PATCHES;

	typedef vector<vector<Mat> > PatchInQ;  // in query
	PatchInQ q_patches;
	typedef vector<vector<Mat> > PatchInC;  // in candidate
	PatchInC c_patches;

	float get_simil(const Mat& a, const Mat& b);  // calculate similarity
};

// notice: the code in "patchmatch.cpp" needs to be modified in case step > 1

#endif
