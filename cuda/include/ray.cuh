#ifndef _RAY_H
#define _RAY_H

#include <vector_functions.h>
#include "vec3.cuh"

class Ray {
public:

	__host__ __device__ Ray(){}
	__host__ __device__ Ray(const point3& _origin, const vec3& _direction)
	:	origin(_origin), direction(_direction) {}

	__host__ __device__  point3 at(double t) const {
		return origin + t * direction;
	}
	point3 origin;
	vec3 direction;
};

#endif