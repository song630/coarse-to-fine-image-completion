#ifndef _PATCHMATCH_H_
#define _PATCHMATCH_H_

#pragma once

#include "patch.h"
#include "segmentation.h"
#include <vector>
#include <string>

#define SIDE_LEN 3

using namespace std;
using namespace cv;

class PatchMatch {
public:
	typedef vector<vector<patch> > PthOfImg;
	typedef vector<vector<Mat> > PatchInQ;

	PatchMatch(const Mat& img, const RECT& _roi, const PthOfImg& pre_patches);
	~PatchMatch();
	void init();
	void propagation_search();
	void print() {
		for (int i = 0; i <= height - SIDE_LEN; i += 10)
		{
			for (int j = 0; j <= width - SIDE_LEN; j += 10)
			{
				cout << "[" << cur_PATCHES[i][j].get_offset().x << ", ";
				cout << cur_PATCHES[i][j].get_offset().y << "]" << endl;
			}
		}
	}

private:
	Mat query;
	int width, height;  // the two images must be of the same size
	static float alpha;  // decay ratio
	static int threshold;  // the similar patch should not be in a nearby region

	// save all patches within the hole, accessed by PthOfImg[][]
	// pre_PATCHES: propagated from last level
	PthOfImg cur_PATCHES, pre_PATCHES;

	PatchInQ q_patches;  // all patches in "query"

	const RECT roi;  // top-left and bottom-right points marking a rectangle
	float get_simil(const Mat& a, const Mat& b);  // calculate similarity
	void color_update();
};

// notice: the code in "patchmatch.cpp" needs to be modified in case step > 1

#endif
