#ifndef _PATCH_H_
#define _PATCH_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class PatchMatch;

class patch {
public:
	patch() {}
	patch(int loc_x, int loc_y) : offset(Point(0, 0)),
		location(Point(loc_x, loc_y)), similarity(0) {}  // initialize

	patch(const Point& point) : offset(Point(0, 0)),
		location(point), similarity(0) {}

	~patch() {}

	void update_offset(const Point& n_offset) {
		offset = n_offset;
	}  // update when finding a new patch that is a better match

	void update_sim(const float sim) {
		similarity = sim;
	}

	float get_sim() {
		return similarity;
	}

	Point get_offset() {
		return offset;
	}

	// propagate info from "pre_PATCHES[]" to "cur_PATCHS[]"
	void propagate(const patch& p, const Point& q, int delta_x, int delta_y) {
		offset.x = p.offset.x + delta_x;
		offset.y = p.offset.y + delta_y;
		location = q;
		similarity = p.similarity;
	}

	friend class PatchMatch;

private:
	// static int side_len;  // size of patch: side_len * side_len
	Point offset;  // from the patch in QUERY IMAGE to patch in CANDIDATE IMAGE
	Point location;  // coordinates of top-left
	float similarity;
};

#endif
