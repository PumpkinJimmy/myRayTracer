#ifndef _COMMON_H
#define _COMMON_H

#include <climits>
#include <random>
#include <vector_functions.h>

#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__

const double inf = std::numeric_limits<double>::infinity();
#define pi 3.141592653589

// Utility Methods

__VECTOR_FUNCTIONS_DECL__  double degree_to_radius(double degree) {
	return degree * pi / 180.0;
}
//
//__VECTOR_FUNCTIONS_DECL__ int random_int(int min, int max) {
//	// int from [min, max]
//	static std::mt19937 gen;
//	std::uniform_int_distribution<> dist(min, max);
//	return dist(gen);
//}
//
//__VECTOR_FUNCTIONS_DECL__ double random_double() {
//	// [0, 1)
//	static std::uniform_real_distribution<double> dist(0.0, 1.0);
//	static std::mt19937 gen;
//	return dist(gen);
//}
//
//__VECTOR_FUNCTIONS_DECL__ inline double random_double(double min_, double max_) {
//	// [min, max)
//	return min_ + (max_ - min_) * random_double();
//}






#endif