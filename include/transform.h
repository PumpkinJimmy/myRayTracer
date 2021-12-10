#pragma once
#include "common.h"
#include "vec3.h"
#include "hittable.h"
class Translate : public Hittable {
public:
	Translate(Hittable::Ptr p, const vec3& displacement)
		:ptr(p), offset(displacement) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override;
public:
	Hittable::Ptr ptr;
	vec3 offset;
};

inline bool Translate::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	ray moved_r(r.origin() - offset, r.direction());
	if (!ptr->hit(moved_r, t_min, t_max, rec)) {
		return false;
	}
	rec.p += offset;
	rec.set_face_normal(moved_r, rec.normal);

	return true;
}
inline bool Translate::bounding_box(double time0, double time1, AABB& output_box) const {
	if (!ptr->bounding_box(time0, time1, output_box))
		return false;

	output_box = AABB(
		output_box.min() + offset,
		output_box.max() + offset);
	return true;
}

class RotateY : public Hittable {
public:
	RotateY(Hittable::Ptr p, double angle);
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
		output_box = bbox;
		return hasbox;
	}
public:
	Hittable::Ptr ptr;
	double sin_theta;
	double cos_theta;
	bool hasbox;
	AABB bbox;
};

inline RotateY::RotateY(Hittable::Ptr p, double angle) : ptr(p) {
	auto radians = degree_to_radius(angle);
	sin_theta = sin(radians);
	cos_theta = cos(radians);
	hasbox = ptr->bounding_box(0, 1, bbox);

	point3 min(inf, inf, inf);
	point3 max(-inf, -inf, -inf);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				auto x = i * bbox.max().x() + (1 - i) * bbox.min().x();
				auto y = j * bbox.max().y() + (1 - j) * bbox.min().y();
				auto z = k * bbox.max().z() + (1 - k) * bbox.min().z();

				auto newx = cos_theta * x + sin_theta * z;
				auto newz = -sin_theta * x + cos_theta * z;

				vec3 tester(newx, y, newz);
				for (int c = 0; c < 3; c++) {
					min[c] = fmin(min[c], tester[c]);
					max[c] = fmax(max[c], tester[c]);
				}
			}
		}
	}
	bbox = AABB(min, max);
}

inline bool RotateY::hit(const ray& r, double t_min, double t_max, hit_record& rec) const{
	auto origin = r.origin();
	auto direction = r.direction();

	origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
	origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
	direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
	direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

	ray rotated_r(origin, direction);

	if (!ptr->hit(rotated_r, t_min, t_max, rec)) {
		return false;
	}

	auto p = rec.p;
	auto normal = rec.normal;

	p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
	p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];
	
	normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
	normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];
	rec.p = p;
	rec.set_face_normal(rotated_r, normal);
}