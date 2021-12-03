#ifndef _VEC3_H
#define _VEC3_H
#include <curand_kernel.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>

#include "cutil_math.h"
#include "common.cuh"

typedef float3 vec3;

typedef float3 color;

typedef float3 point3;

__VECTOR_FUNCTIONS_DECL__ point3 make_point3(float x, float y, float z) {
	return make_float3(x, y, z);
}

__VECTOR_FUNCTIONS_DECL__ color make_color(float x, float y, float z) {
	return make_float3(x, y, z);
}

__VECTOR_FUNCTIONS_DECL__ vec3 make_vec3(float x, float y, float z) {
	return make_float3(x, y, z);
 }


__VECTOR_FUNCTIONS_DECL__ vec3 unit_vector(vec3 v) {
	 return normalize(v);
}

__VECTOR_FUNCTIONS_DECL__ float length_squared(vec3 v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__VECTOR_FUNCTIONS_DECL__ float3 operator-(float3 v) {
	return make_float3(-v.x, -v.y, -v.z);
}

__VECTOR_FUNCTIONS_DECL__ bool near_zero(float3 v) {
	const float eps = 1e-5;
	return fabsf(v.x) < eps && fabsf(v.y) < eps && fabsf(v.z) < eps;
}

__device__ inline vec3 random_vec3(curandState* randState, float min_, float max_) {
	return make_vec3(random_real(randState, min_, max_), random_real(randState, min_, max_), random_real(randState, min_, max_));
}

__device__ inline vec3 random_in_unit_sphere(curandState* randState) {
	while (true) {
		auto p = random_vec3(randState, -1, 1);
		if (length_squared(p) >= 1) continue;
		return p;
	}
}
__device__ inline vec3 random_unit_vector(curandState* randState) {
	return unit_vector(random_in_unit_sphere(randState));
}

#endif