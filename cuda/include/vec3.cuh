#ifndef _VEC3_H
#define _VEC3_H
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


#endif