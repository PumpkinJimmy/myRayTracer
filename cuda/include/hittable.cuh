#pragma once
#ifndef _HITTABLE_H
#define _HITTABLE_H

#include <memory>
#include "ray.cuh"
#include "common.cuh"
#include "vec3.cuh"

class Material;

struct hit_record {
	point3 p;
	vec3 normal;
	float t;
	float u;
	float v;
	Material* mat_ptr;
	bool front_face;

	__host__ __device__ inline void set_face_normal(const Ray& r, vec3 outward_normal) {
		front_face = dot(r.direction, outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hittable {
public:
	__host__ __device__ virtual bool hit(
		const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	typedef Hittable* Ptr;
	typedef const Hittable* ConstPtr;
};

#endif