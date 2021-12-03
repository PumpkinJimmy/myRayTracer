#ifndef _MODEL_H
#define _MODEL_H
#include <cuda_runtime.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
struct SphereData {
	point3 cen;
	float r;
};
class Sphere {
public:
	__host__ __device__ Sphere() {}
	__host__ __device__ Sphere(point3 cen, float r)
		:center(cen), radius(r) {}

	__host__ __device__ Sphere(SphereData d)
		:center(d.cen), radius(d.r){}

	__host__ __device__ bool hit(
		const Ray& r, float t_min, float t_max, hit_record& rec) const {
		vec3 oc = r.origin - center;
		float a = length_squared(r.direction);
		float half_b = dot(oc, r.direction);
		float c = length_squared(oc) - radius * radius;

		float delta = half_b * half_b - a * c;
		
		if (delta < 0) { 
			return false; 
		}
		
		auto sqrtd = sqrt(delta);

		auto root = (-half_b - sqrtd) / a;
		if (root < t_min || t_max < root) {
			root = (-half_b + sqrtd) / a;
			if (root < t_min || t_max < root)
				printf("False2\n");
				return false;
		}
		

		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;
		rec.set_face_normal(r, outward_normal);
		return true;
	}

	
public:
	point3 center;
	float radius;
};

#endif