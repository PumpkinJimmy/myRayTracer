#pragma once
#ifndef _BOX_H
#define _BOX_H

#include "common.h"
#include "aarect.h"
#include "material.h"
#include "vec3.h"

#include "hittable_list.h"

class Box : public Hittable {
public:
	Box() {}
	Box(const point3& p0, const point3& p1, Material::Ptr ptr);

	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
		output_box = AABB(box_min, box_max);
		return true;
	}
public:
	point3 box_min;
	point3 box_max;
	HittableList sides;
};

Box::Box(const point3& p0, const point3& p1, Material::Ptr ptr) {
	box_min = p0;
	box_max = p1;

	sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
	sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));
	sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
	sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));
	sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
	sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool Box::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	return sides.hit(r, t_min, t_max, rec);
}

#endif