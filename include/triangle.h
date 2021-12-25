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
	typedef shared_ptr<Vertex> Ptr;
};
class Triangle : public Hittable {
public:
	typedef shared_ptr<Triangle> Ptr;
	Triangle() = default;
	
	Triangle(Vertex::Ptr v0, Vertex::Ptr v1, Vertex::Ptr v2, Material::Ptr m) {
		mat_ptr = m;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e1 = vertices[1]->position - vertices[0]->position;
		e2 = vertices[2]->position - vertices[0]->position;
		tri_normal = calNormal(v0, v1, v2);
		setAABB();
		hasNorm = false;
	}
	Triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, Material::Ptr m)
		: Triangle(shared_ptr<Vertex>(new Vertex(v0)), shared_ptr<Vertex>(new Vertex(v1)), shared_ptr<Vertex>(new Vertex(v2)), m) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
		double b1, b2;
		if (!rayIntersect(r, rec.t, b1, b2) || rec.t < t_min || rec.t > t_max) {
			return false;
		}
		
		rec.p = r.at(rec.t);
		rec.mat_ptr = mat_ptr;
		auto uv = (1 - b1 - b2) * vertices[0]->tex_coord + b1 * vertices[1]->tex_coord + b2 * vertices[2]->tex_coord;
		rec.u = uv[0]; rec.v = uv[1];
		if (hasNorm) {
			rec.set_face_normal(r, tri_normal);
			rec.normal = (1 - b1 - b2) * vertices[0]->normal + b1 * vertices[1]->normal + b2 * vertices[2]->normal;
		}
		else {
			rec.set_face_normal(r, tri_normal);
		}
		if (!rec.front_face) {
			return false;
		}
		return true;
		
	}
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const {
		output_box = bbox;
		return true;
	}
	void setMaterial(Material::Ptr mat) {
		mat_ptr = mat;
	}

	static vec3 calNormal(Vertex::Ptr v0, Vertex::Ptr v1, Vertex::Ptr v2) {
		return normalize(cross(v1->position - v0->position, v2->position - v0->position));
	}

	template <typename... Args>
	static Triangle::Ptr create(Args... args) {
		return make_shared<Triangle>(args...);
	}

	bool hasNorm;
private:
	Vertex::Ptr vertices[3];
	vec3 tri_normal;
	Material::Ptr mat_ptr;
	AABB bbox;
	vec3 e1;
	vec3 e2;
	bool rayIntersect(const ray& r, double& t, double& u, double& v) const {
		
		vec3 tvec = r.orig - vertices[0]->position;
		vec3 pvec = cross(r.dir, e2);
		auto det = dot(e1, pvec);

		det = 1.0 / det;

		u = dot(tvec, pvec) * det;

		if (isnan(u)){
			printf("Ray in triangle surface");
		}

		if (u < 0.0 || u > 1.0 || isnan(u)) {
			return false;
		}

		vec3 qvec = cross(tvec, e1);

		v = dot(r.dir, qvec) * det;

		if (v < 0.0 || (u + v) > 1.0 || isnan(u)) {
			return false;
		}

		t = dot(e2, qvec) * det;

		return true;
	}
	void setAABB() {
		vec3 min_p = vertices[0]->position, max_p = vertices[0]->position;
		for (int i = 1; i < 3; i++) {
			Vertex::Ptr v = vertices[i];
			for (int j = 0; j < 3; j++) {
				min_p[j] = std::min(min_p[j], v->position[j]);
				max_p[j] = std::max(max_p[j], v->position[j]);
			}
		}
		min_p = min_p - vec3(0.0001, 0.0001, 0.0001);
		max_p = max_p + vec3(0.0001, 0.0001, 0.0001);
		bbox = AABB(min_p, max_p);
	}
};

#endif