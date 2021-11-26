#pragma once
#ifndef _HITTABLE_H
#define _HITTABLE_H

#include "ray.h"
#include "common.h"
#include "aabb.h"

class Material;

struct hit_record {
	point3 p;
	vec3 normal;
	double t;
	shared_ptr<Material> mat_ptr;

	bool front_face;

	inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hittable {
public:
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const = 0;
	typedef shared_ptr<Hittable> Ptr;
	typedef shared_ptr<const Hittable> ConstPtr;
};

#endif