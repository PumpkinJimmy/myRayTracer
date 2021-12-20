#pragma once
#ifndef _COMMON_H
#define _COMMON_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <random>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double inf = std::numeric_limits<double>::infinity();
const double pi = 3.141592653589;

// Utility Methods

inline double degree_to_radius(double degree) {
	return degree * pi / 180.0;
}

inline int random_int(int min, int max) {
	// int from [min, max]
	extern int seed;
	static std::mt19937 gen(seed);
	std::uniform_int_distribution<> dist(min, max);
	return dist(gen);
}

inline double random_double() {
	// [0, 1)
	extern int seed;
	static std::uniform_real_distribution<double> dist(0.0, 1.0);
	static std::mt19937 gen(seed);
	return dist(gen);
}

inline double random_double(double min_, double max_) {
	// [min, max)
	return min_ + (max_ - min_) * random_double();
}

inline double clamp(double x, double min, double max) {
	if (x > max) return max;
	if (x < min) return min;
	return x;
}



// Common Libs

#include <memory>

#endif