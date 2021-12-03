#ifndef _MATERIAL_H
#define _MATERIAL_H
#include <curand_kernel.h>
#include "common.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "hittable.cuh"

class Material {
public:

	__device__ virtual bool scatter(
		const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* randState
	) const = 0;

	typedef Material* Ptr;
	typedef const Material* ConstPtr;

};

class Lambertian : public Material {
public:
	__device__ Lambertian(const color& a) : albedo(a) {}

	__device__ virtual bool scatter(
		const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* randState
	) const override {
		//auto scatter_direction = rec.normal + random_unit_vector(randState);
		auto scatter_direction = rec.normal + vec3{ 0, 0, 1 };

		if (near_zero(scatter_direction)) {
			scatter_direction = rec.normal;
		}
		scattered = Ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}
	template <typename... Args>
	__device__ static Lambertian::Ptr create(Args... args) {
		return new Lambertian(args...);
		// return make_shared<Lambertian>(args...);
	}
public:
	color albedo;
};

#endif