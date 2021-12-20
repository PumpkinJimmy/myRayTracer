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

__device__ int bytecmp(const uint8_t* src, const uint8_t* dst, int len) {
	for (int i = 0; i < len; i++) {
		if (src[i] != dst[i]) return src[i] > dst[i] ? 1 : -1;
	}
	return 0;
}

__device__ void bytecpy(uint8_t* dst, const uint8_t* src, int len) {
	for (int i = 0; i < len; i++) {
		dst[i] = src[i];
	}
}

#endif