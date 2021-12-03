#ifndef _COMMON_H
#define _COMMON_H

#include <climits>
#include <random>
#include <vector_functions.h>
#include <cutil_math.h>
#include <curand_kernel.h>

#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__

const double inf = std::numeric_limits<double>::infinity();
#define pi 3.141592653589

// Utility Methods

__VECTOR_FUNCTIONS_DECL__  double degree_to_radius(double degree) {
	return degree * pi / 180.0;
}

int quantize(float f) {
	return static_cast<int>(f * 255.999);
}
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ float random_real(curandState* randState) {
	// [0, 1]
	return curand_uniform(randState);
}

__device__ float random_real(curandState* randState, float a, float b) {
	// [a, b]
	return a + (b - a) * random_real(randState);
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