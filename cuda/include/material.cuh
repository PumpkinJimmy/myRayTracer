#ifndef _MATERIAL_H
#define _MATERIAL_H
#include <curand_kernel.h>
#include <cutil_math.h>
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
		auto scatter_direction = rec.normal + random_unit_vector(randState);
		//auto scatter_direction = rec.normal + vec3{ 0, 0, 1 };

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


class Metal : public Material {
public:
	__device__ Metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ virtual bool scatter(
		const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* randState
	) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(randState));
		attenuation = albedo;
		return (dot(scattered.direction, rec.normal) > 0);
	}
	template<typename... Args>
	__device__ static Metal::Ptr create(Args... args) {
		return new Metal(args...);
	}
public:
	color albedo;
	float fuzz;
};

class Dielectric : public Material {
public:
	__device__  Dielectric(float index_of_refraction) : ir(index_of_refraction) {}
	template<typename... Args>
	__device__  static Dielectric::Ptr create(Args... args) {
		return new Dielectric(args...);
	}
	__device__ virtual bool scatter(
		const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* randState
	) const override {
		attenuation = color{ 1.0, 1.0, 1.0 };
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction);
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_real(randState)) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = Ray(rec.p, direction);
		return true;
	}


public:
	float ir;
private:
	__device__  static double reflectance(double cosine, double ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};


#endif