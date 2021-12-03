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

	__device__ virtual bool serialize(uint8_t*& start, const uint8_t* end) const = 0;

	typedef Material* Ptr;
	typedef const Material* ConstPtr;
};

class Lambertian : public Material {
public:
	struct LambertianData {
		color a;
	};
	__device__ Lambertian(const color& a) : albedo(a) {}
	__device__ Lambertian(const LambertianData& d) : Lambertian(d.a) {}
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

	__device__ virtual bool serialize(uint8_t*& start, const uint8_t* end) const override {
		if (end - start < 4 + sizeof(LambertianData)) return false;
		bytecpy(start, "LAMB", 4);
		start += 4;
		LambertianData d{ albedo };
		bytecpy(start, (uint8_t*)&d, sizeof(LambertianData));
		start += sizeof(LambertianData);
		return true;
	}

	template <typename... Args>
	__device__ static Lambertian::Ptr create(Args... args) {
		return new Lambertian(args...);
		// return make_shared<Lambertian>(args...);
	}
	__device__ static bool deserialize(uint8_t*& start, const uint8_t* end, Material::Ptr& res) {
		if (end - start < sizeof(LambertianData)) {
			printf("Deserialize Lambertian failed\n");
			return false;
		}
		res = new Lambertian(*((LambertianData*)start));
		start += sizeof(LambertianData);
		return true;
	}
	
public:
	color albedo;
};


class Metal : public Material {
public:
	struct MetalData {
		color a;
		float f;
	};
	__device__ Metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ Metal(const MetalData& d) : Metal(d.a, d.f) {}
	__device__ virtual bool scatter(
		const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* randState
	) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(randState));
		attenuation = albedo;
		return (dot(scattered.direction, rec.normal) > 0);
	}

	__device__ virtual bool serialize(uint8_t*& start, const uint8_t* end) const override {
		if (end - start < 4 + sizeof(MetalData)) return false;
		bytecpy(start, "META", 4);
		start += 4;
		MetalData d{ albedo, fuzz };
		bytecpy(start, (uint8_t*)&d, sizeof(MetalData));
		start += sizeof(MetalData);
		return true;
	}

	template<typename... Args>
	__device__ static Metal::Ptr create(Args... args) {
		return new Metal(args...);
	}
	__device__ static bool deserialize(uint8_t*& start, const uint8_t* end, Material::Ptr& res) {
		if (end - start < sizeof(MetalData)) {
			printf("Deserialize Metal failed\n");
			return false;
		}
		res = new Metal(*((MetalData*)start));
		start += sizeof(MetalData);
		return true;
	}

public:
	color albedo;
	float fuzz;
};

class Dielectric : public Material {
public:
	struct DielectricData {
		float index_of_refraction;
	};
	__device__ Dielectric(float index_of_refraction) : ir(index_of_refraction) {}
	__device__ Dielectric(const DielectricData& d) : Dielectric(d.index_of_refraction) {}
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
		double cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
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

	__device__ virtual bool serialize(uint8_t*& start, const uint8_t* end) const override {
		if (end - start < 4 + sizeof(DielectricData)) return false;
		bytecpy(start, "DIEL", 4);
		start += 4;
		DielectricData d{ ir };
		bytecpy(start, (uint8_t*)&d, sizeof(DielectricData));
		start += sizeof(DielectricData);
		return true;
	}

	__device__ static bool deserialize(uint8_t*& start, const uint8_t* end, Material::Ptr& res) {
		if (end - start < sizeof(DielectricData)) {
			printf("Deserialize Dielectric failed\n");
			return false;
		}
		res = new Dielectric(*((DielectricData*)start));
		start += sizeof(DielectricData);
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

__device__ static bool material_deserialize(uint8_t*& start, const uint8_t* end, Material::Ptr& res) {
	const int len = 4;
	if (end - start < len) return false;
	uint8_t mat_type[len];
	for (int i = 0; i < len && start + i != end; i++) {
		mat_type[i] = start[i];
	}
	start += len;
	if (bytecmp(mat_type, "LAMB", len) == 0) {
		return Lambertian::deserialize(start, end, res);
	}
	else if (bytecmp(mat_type, "META", len) == 0) {
		return Metal::deserialize(start, end, res);
	}
	else if (bytecmp(mat_type, "DIEL", len) == 0) {
		return Dielectric::deserialize(start, end, res);
	}
	else return false;
}


#endif