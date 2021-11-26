#pragma once
#ifndef _SPHERE_H
#define _SPHERE_H

#include "common.h"
#include "hittable.h"
#include "material.h"

class Sphere : public Hittable {
public:
	Sphere() {}
	Sphere(point3 cen, double r, Material::Ptr m) : 
		center(cen), radius(r), mat_ptr(m) {};
	virtual bool hit(
		const ray& r, double t_min, double t_max, hit_record& rec) const override;
	
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
		output_box = AABB(
			center - vec3(radius, radius, radius),
			center + vec3(radius, radius, radius)
			);
		return true;
	}
	
	typedef shared_ptr<Sphere> Ptr;
	typedef shared_ptr<const Sphere> ConstPtr;
	
	template <typename... Args>
	static Sphere::Ptr create(Args... args) {
		return make_shared<Sphere>(args...);
	}
private:
	point3 center;
	double radius;
	Material::Ptr mat_ptr;
};

bool Sphere::hit(
	const ray& r, double t_min, double t_max, hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;

	auto delta = half_b * half_b - a * c;
	if (delta < 0) return false;

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
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;
}
#endif