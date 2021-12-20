#pragma once
#ifndef _HITTABLE_LIST_H
#define _HITTABLE_LIST_H

#include "common.cuh"
#include "hittable.cuh"
//
//class HittableList : public Hittable {
//public:
//	HittableList() {}
//
//	typedef shared_ptr<HittableList> Ptr;
//	typedef shared_ptr<const HittableList> ConstPtr;
//	static Ptr create() {
//		return make_shared<HittableList>();
//	}
//
//	void add(Hittable::ConstPtr obj) {
//		objects.push_back(obj);
//	}
//	void clear() { objects.clear(); }
//	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
//	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override {
//		if (objects.empty()) return false;
//
//		AABB tmp_box;
//		bool first_box = true;
//
//		for (const auto& object : objects) {
//			if (!object->bounding_box(time0, time1, tmp_box)) return false;
//			output_box = first_box ? tmp_box : surrounding_box(output_box, tmp_box);
//			first_box = false;
//		}
//		return true;
//	}
//public:
//	vector<Hittable::ConstPtr> objects;
//};
//
//bool HittableList::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
//{
//	hit_record tmp;
//	bool hit_any = false;
//	auto cur_closest = t_max;
//
//	for (const auto& obj : objects) {
//		if (obj->hit(r, t_min, cur_closest, tmp)) {
//			hit_any = true;
//			cur_closest = tmp.t;
//			rec = tmp;
//		}
//	}
//	return hit_any;
//}
#endif