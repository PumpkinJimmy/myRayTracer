#ifndef _MODEL_H
#define _MODEL_H
#include <cuda_runtime.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "material.cuh"
struct SphereData {
	point3 cen;
	float r;
};
class Sphere: public Hittable {
public:
	typedef Sphere* Ptr;
	typedef const Sphere* ConstPtr;
	__host__ __device__ Sphere() {}
	__host__ __device__ Sphere(point3 cen, float r, Material::Ptr mat)
		:center(cen), radius(r), mat_ptr(mat) {}

	__host__ __device__ Sphere(SphereData d)
		: Sphere(d.cen, d.r, nullptr) {}

	__device__ bool hit(
		const Ray& r, float t_min, float t_max, hit_record& rec) const {
		auto oc = float3(r.origin) - float3(center);
		auto rd = r.direction;
		auto a = length_squared(rd);
		auto half_b = dot(oc, rd);
		auto c = length_squared(oc) - double(radius) * radius;

		auto delta = half_b * half_b - a * c;

		if (delta < 0) {
			return false;
		}

		auto sqrtd = sqrt(delta);

		auto root = (-half_b - sqrtd) / a;
		if (root < t_min || t_max < root) {
			root = (-half_b + sqrtd) / a;
			if (root < t_min || t_max < root)
				return false;
		}


		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;
		rec.mat_ptr = mat_ptr;
		rec.set_face_normal(r, outward_normal);
		return true;
	}
	__device__ virtual bool serialize(uint8_t*& start, const uint8_t* end) const {
		
		SphereData d{ center, radius };
		*(SphereData*)(start) = d;
		start += sizeof(SphereData);
		if (!mat_ptr->serialize(start, end)) return false;
		return true;
	}

	__device__ static bool deserialize(uint8_t*& start, const uint8_t* end, Sphere*& res) {
		if (end - start < sizeof(SphereData)) {
			return false;
		}
		else {
			SphereData d = *(SphereData*)start;
			start += sizeof(SphereData);
			Material::Ptr mat;
			if (material_deserialize(start, end, mat)) {
				res = new Sphere(d.cen, d.r, mat);
				return true;
			}
			else return false;
		}
	}
	template<typename... Args>
	__device__ static Sphere::Ptr create(Args... args) {
		return new Sphere(args...);
	}


public:
	point3 center;
	float radius;
	Material::Ptr mat_ptr;
};

#endif