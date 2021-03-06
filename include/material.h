#pragma once
#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "common.h"
#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "texture.h"

class Material {
public:
	virtual color emitted(double u, double v, const point3& p) const {
		return color(0, 0, 0);
	}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const = 0;

	typedef shared_ptr<Material> Ptr;
	typedef shared_ptr<const Material> ConstPtr;

};

class Lambertian : public Material {
public:
	Lambertian(const color& a) : albedo(make_shared<solid_color>(a)) {}
	Lambertian(shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override {
		// if (!rec.front_face) return false;
		auto scatter_direction = rec.normal + random_unit_vector();

		if (scatter_direction.near_zero()) {
			scatter_direction = rec.normal;
		}
		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}
	template <typename... Args>
	static Lambertian::Ptr create(Args... args) {
		return make_shared<Lambertian>(args...);
	}
public:
	shared_ptr<texture> albedo;
};

class Metal : public Material {
public:
	Metal(const color& a, double f) : albedo(make_shared<solid_color>(a)), fuzz(f < 1 ? f : 1) {}
	Metal(shared_ptr<texture> tex, double f) : albedo(tex), fuzz(f < 1 ? f : 1) {}
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override {
		// if (!rec.front_face) return false;
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return (dot(scattered.direction(), rec.normal) > 0);
	}
	template<typename... Args>
	static Metal::Ptr create(Args... args) {
		return make_shared<Metal>(args...);
	}
public:
	shared_ptr<texture> albedo;
	double fuzz;
};

class Dielectric : public Material {
public:
	Dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	template<typename... Args>
	static Dielectric::Ptr create(Args... args) {
		return make_shared<Dielectric>(args...);
	}
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override {
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = ray(rec.p, direction);
		return true;
	}


public:
	double ir;

private:
	static double reflectance(double cosine, double ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};


class diffuse_light : public Material {
public:
	diffuse_light(shared_ptr<texture> a) : emit(a) {}
	diffuse_light(color c) : emit(make_shared<solid_color>(c)) {}
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray&
		scattered
	) const override {
		return false;
	}
	virtual color emitted(double u, double v, const point3& p) const
		override {
		return emit->value(u, v, p);
	}
public:
	shared_ptr<texture> emit;
};

class Plastic : public Material {
public:

	Plastic(shared_ptr<texture> tex, double index_of_refraction, double rough=0.01) : albedo(tex), ir(index_of_refraction), roughness(rough){}
	Plastic(color c, double index_of_refraction, double rough=0.01) : albedo(make_shared<solid_color>(c)), ir(index_of_refraction), roughness(rough) {}
	template<typename... Args>
	static Plastic::Ptr create(Args... args) {
		return make_shared<Plastic>(args...);
	}
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override {
		
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		vec3 direction;

		double R = reflectance(cos_theta, refraction_ratio);

		if (R > random_double()) {
			direction = reflect(unit_direction, rec.normal) + roughness * random_unit_vector();
			attenuation = albedo->value(rec.u, rec.v, rec.p);
		}
		else {
			direction = rec.normal + random_unit_vector();
			attenuation = albedo->value(rec.u, rec.v, rec.p);

		}

		scattered = ray(rec.p, direction);
		return true;
	}


public:
	shared_ptr<texture> albedo;
	double ir;
	double roughness;

private:
	static double reflectance(double cosine, double ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};

#endif