#pragma once
#ifndef _TRIANGLE_H
#define _TRIANGLE_H
#include "vec3.h"
#include "hittable.h"
#include "aabb.h"
#include "material.h"
struct Vertex {
	point3 position;
	vec3 normal;
	vec3 tex_coord;
};
class Triangle : public Hittable {
public:
	typedef shared_ptr<Triangle> Ptr;
	Triangle() = default;
	Triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, Material::Ptr m) {
		mat_ptr = m;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		tri_normal = calNormal(v0, v1, v2);
		setAABB();
	}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
		double b1, b2;
		if (!rayIntersect(r, rec.t, b1, b2) || rec.t < t_min || rec.t > t_max) {
			return false;
		}
		rec.p = r.at(rec.t);
		rec.mat_ptr = mat_ptr;
		auto uv = (1 - b1 - b2) * vertices[0].tex_coord + b1 * vertices[1].tex_coord + b2 * vertices[2].tex_coord;
		rec.u = uv[0]; rec.v = uv[1];
		rec.normal = (1 - b1 - b2) * vertices[0].normal + b1 * vertices[1].normal + b2 * vertices[2].normal;
		rec.set_face_normal(r, tri_normal);
		return true;
		
	}
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const {
		output_box = bbox;
		return true;
	}

	static vec3 calNormal(Vertex v0, Vertex v1, Vertex v2) {
		return normalize(cross(v2.position - v0.position, v1.position - v0.position));
	}

	template <typename... Args>
	static Triangle::Ptr create(Args... args) {
		return make_shared<Triangle>(args...);
	}
private:
	Vertex vertices[3];
	vec3 tri_normal;
	Material::Ptr mat_ptr;
	AABB bbox;
	bool rayIntersect(const ray& r, double& t, double& u, double& v) const {
		vec3 e1 = vertices[1].position - vertices[0].position;
		vec3 e2 = vertices[2].position - vertices[0].position;
		vec3 tvec = r.orig - vertices[0].position;
		vec3 pvec = cross(r.dir, e2);
		auto det = dot(e1, pvec);

		det = 1.0 / det;

		u = dot(tvec, pvec) * det;

		if (u < 0.0 || u > 1.0) {
			return false;
		}

		vec3 qvec = cross(tvec, e1);

		v = dot(r.dir, qvec) * det;

		if (v < 0.0 || (u + v) > 1.0) {
			return false;
		}

		t = dot(e2, qvec) * det;

		return true;
	}
	void setAABB() {
		vec3 min_p = vertices[0].position, max_p = vertices[0].position;
		for (int i = 1; i < 3; i++) {
			const Vertex& v = vertices[i];
			for (int j = 0; j < 3; j++) {
				min_p[j] = std::min(min_p[j], v.position[j]);
				max_p[j] = std::max(max_p[j], v.position[j]);
			}
		}
		min_p = min_p - vec3(0.0001, 0.0001, 0.0001);
		max_p = max_p + vec3(0.0001, 0.0001, 0.0001);
		bbox = AABB(min_p, max_p);
	}
};

#endif