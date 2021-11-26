#pragma once
#ifndef _BVH_H
#define _BVH_H

#include <vector>
#include <functional>
#include "hittable.h"
#include "hittable_list.h"


inline bool box_compare(Hittable::ConstPtr a, Hittable::ConstPtr b, int axis) {
	AABB box_a;
	AABB box_b;

	if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
		std::cerr << "No bounding box in BVHNode constructor. \n";
	}


	return box_a.min().e[axis] < box_b.min().e[axis];
}

class BVHNode : public Hittable {
public:
	BVHNode() {}
	BVHNode(
		const std::vector<Hittable::ConstPtr>& src_objects,
		size_t start, size_t end, double time0, double time1
	);
	BVHNode(const HittableList& list, double time0, double time1) :
		BVHNode(list.objects, 0, list.objects.size(), time0, time1) {}
	virtual bool hit(
		const ray& r, double t_min, double t_max, hit_record& rec) const;
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const override
	{
		output_box = box;
		return true;
	}
	template <typename... Args>
	auto create(Args... args) {
		return make_shared<BVHNode>(args...);
	}
public:
	Hittable::ConstPtr left;
	Hittable::ConstPtr right;
	AABB box;
};

inline bool BVHNode::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	if (!box.hit(r, t_min, t_max))
		return false;
	bool hit_left = left->hit(r, t_min, t_max, rec);
	bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

	return hit_left || hit_right;
}

BVHNode::BVHNode(
	const std::vector<Hittable::ConstPtr>& src_objects,
	size_t start, size_t end, double time0, double time1
) {
	auto objects = src_objects;

	auto axis = random_int(0, 2);
	
	auto comparator = std::bind(box_compare, std::placeholders::_1, std::placeholders::_2, axis);

	size_t object_span = end - start;

	if (object_span == 1) {
		left = right = objects[start];
	}
	else if (object_span == 2) {
		if (comparator(objects[start], objects[start + 1])) {
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			left = objects[start + 1];
			right = objects[start];
		}
	}
	else {
		std::sort(objects.begin() + start, objects.begin() + end, comparator);

		auto mid = start + object_span / 2;
		left = BVHNode::create(objects, start, mid, time0, time1);
		right = BVHNode::create(objects, mid, end, time0, time1);
	}

	AABB box_left, box_right;

	if (!left->bounding_box(time0, time1, box_left)
		|| !right->bounding_box(time0, time1, box_right)
		)
		std::cerr << "No bounding box in BVHNode constructor. \n";


	box = surrounding_box(box_left, box_right);
}


#endif // !_BVH_H
