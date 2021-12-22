#pragma once
#ifndef _SCENE_H
#define _SCENE_H
#include "common.h"
#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"
#include "material.h"
#include "texture.h"
#include "aarect.h"
#include "box.h"
#include "transform.h"
#include "triangle.h"


HittableList random_scene_simple() {
	auto world = HittableList();
	auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
		color(0.9, 0.9, 0.9));
	auto solid = make_shared<solid_color>(color(0.5, 0.5, 0.5));
	auto ground_meterial = Lambertian::create(solid);
	world.add(Sphere::create(point3(0, -1000, 0), 1000, ground_meterial));

	for (int a = -2; a < 2; a++) {
		for (int b = -2; b < 2; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				Material::Ptr sphere_material;
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = Lambertian::create(albedo);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = Metal::create(albedo, fuzz);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = Dielectric::create(1.5);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = Dielectric::create(1.5);
	world.add(Sphere::create(point3(0, 1, 0), 1.0, material1));

	auto material2 = Lambertian::create(color(0.4, 0.2, 0.1));
	world.add(Sphere::create(point3(-4, 1, 0), 1.0, material2));


	auto material3 = Metal::create(color(0.7, 0.6, 0.5), 0.0);
	world.add(Sphere::create(point3(4, 1, 0), 1.0, material3));

	return world;
}

HittableList random_scene() {
	auto world = HittableList();
	auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
		color(0.9, 0.9, 0.9));
	auto solid = make_shared<solid_color>(color(0.5, 0.5, 0.5));
	auto ground_meterial = Lambertian::create(checker);
	world.add(Sphere::create(point3(0, -1000, 0), 1000, ground_meterial));
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				Material::Ptr sphere_material;
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = Lambertian::create(albedo);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = Metal::create(albedo, fuzz);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = Dielectric::create(1.5);
					world.add(Sphere::create(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = Dielectric::create(1.5);
	world.add(Sphere::create(point3(0, 1, 0), 1.0, material1));

	auto material2 = Lambertian::create(color(0.4, 0.2, 0.1));
	world.add(Sphere::create(point3(-4, 1, 0), 1.0, material2));


	auto material3 = Metal::create(color(0.7, 0.6, 0.5), 0.0);
	world.add(Sphere::create(point3(4, 1, 0), 1.0, material3));

	return world;
}

HittableList simple_light() {
	HittableList objects;
	/*auto pertext = make_shared<texture>(4);
	objects.add(make_shared<Sphere>(point3(0, -1000, 0), 1000,
		make_shared<Lambertian>(pertext)));
	objects.add(make_shared<Sphere>(point3(0, 2, 0), 2, make_shared<Lambertian>
		(pertext)));*/
	auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
		color(0.9, 0.9, 0.9));
	objects.add(Sphere::create(point3(0, -1000, 0), 1000, Lambertian::create(checker)));
	objects.add(Sphere::create(point3(0, 2, 0), 2, Lambertian::create(checker)));
	auto difflight = make_shared<diffuse_light>(color(4, 4, 4));
	objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));
	return objects;
}

HittableList cornell_box() {
	HittableList objects;
	auto red = make_shared<Lambertian>(color(.65, .05, .05));
	auto white = make_shared<Lambertian>(color(.73, .73, .73));
	auto green = make_shared<Lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuse_light>(color(15, 15, 15));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(make_shared<xz_rect>(213, 343, 227, 332, 554, light));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

	Hittable::Ptr box1 = make_shared<Box>(
		point3(0, 0, 0), point3(165, 330, 165), white);
	box1 = make_shared<RotateY>(box1, 15);
	box1 = make_shared<Translate>(box1, vec3(265, 0, 295));
	objects.add(box1);

	Hittable::Ptr box2 = make_shared<Box>(
		point3(0, 0, 0), point3(165, 165, 165), white);
	box2 = make_shared<RotateY>(box2, -18);
	box2 = make_shared<Translate>(box2, vec3(130, 0, 65));
	objects.add(box2);
	return objects;
}

HittableList earth() {
	// auto earth_texture = ImageTexture::create("../../assets/earthmap.jpg");
	auto earth_texture = ImageTexture::create("E:/CG_ws/project/myRayTracer/assets/earthmap.jpg");
	auto earth_surface = Lambertian::create(earth_texture);
	auto globe = Sphere::create(point3(0, 0, 0), 2, earth_surface);

	HittableList objects; objects.add(globe);
	return objects;
}

HittableList final_scene2() {
	HittableList boxes1;
	auto ground = Lambertian::create(color(0.48, 0.83, 0.53));

	const int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++) {
			auto w = 100.0;
			auto x0 = -1000.0 + i * w;
			auto z0 = -1000.0 + j * w;
			auto y0 = 0.0;
			auto x1 = x0 + w;
			auto y1 = random_double(1, 101);
			auto z1 = z0 + w;

			boxes1.add(make_shared<Box>(point3(x0, y0, z0), point3(x1, y1, z1), ground));
		}
	}

	HittableList objects;

	objects.add(make_shared<BVHNode>(boxes1, 0, 1));

	auto light = make_shared<diffuse_light>(color(7, 7, 7));
	objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

	auto center1 = point3(400, 400, 200);
	auto center2 = center1 + vec3(30, 0, 0);

	auto moving_sphere_material = Lambertian::create(color(0.7, 0.3, 0.1));
	objects.add(Sphere::create(center1, 50, moving_sphere_material));
	objects.add(Sphere::create(point3(260, 150, 45), 50, Dielectric::create(1.5)));
	objects.add(Sphere::create(
		point3(0, 150, 145),
		50,
		Metal::create(color(0.8, 0.8, 0.9), 1.0)));

	auto boundary = Sphere::create(
		point3(360, 150, 145), 70, Dielectric::create(1.5));
	objects.add(boundary);
	//boundary = Sphere::create(
	//	point3(0, 0, 0), 5000, Dielectric::create(1.5));

	auto emat = Lambertian::create(
		ImageTexture::create(
			"E:\\CG_ws\\project\\myRayTracer\\assets\\earthmap.jpg"));

	objects.add(Sphere::create(point3(400, 200, 400), 100, emat));


	objects.add(
		Sphere::create(
			point3(220, 280, 300), 80, Lambertian::create(make_shared<solid_color>(color(0.2, 0.4, 0.9))))
	);

	HittableList boxes2;
	auto white = Lambertian::create(color(.73, .73, .73));

	int ns = 1000;

	for (int j = 0; j < ns; j++) {
		boxes2.add(Sphere::create(point3::random(0, 165), 10, white));
	}

	objects.add(make_shared<Translate>(
		make_shared<RotateY>(
			make_shared<BVHNode>(boxes2, 0, 1), 15),
		vec3(-100, 270, 395)
		)
	);
	return objects;

}

HittableList simple_triangle() {
	HittableList objects;
	auto mat = Lambertian::create(make_shared<solid_color>(color(0, 0, 1)));
	Vertex v0{ {0.5, 0, -1}, {0, 0, -1}, {1, 0, 0} };
	Vertex v1{ {-0.5, 0, -1}, {0, 0, -1}, {0, 0, 0} };
	Vertex v2{ {0, 0.5, -1}, {0, 0, -1}, {0, 1, 0} };
	auto tri = Triangle::create(v0, v1, v2, mat);
	objects.add(tri);
	return objects;
}

HittableList simple_triangle2() {
	HittableList objects;
	auto mat = Lambertian::create(make_shared<ImageTexture>("E:/CG_ws/project/myRayTracer/assets/staircase2/textures/wood5.tga"));
	Vertex v0{ {0.5, 0, -1}, {0, 0, -1}, {1, 0, 0} };
	Vertex v1{ {-0.5, 0, -1}, {0, 0, -1}, {0, 0, 0} };
	Vertex v2{ {0, 0.5, -1}, {0, 0, -1}, {0, 1, 0} };
	auto tri = Triangle::create(v0, v1, v2, mat);
	objects.add(tri);
	return objects;
}

HittableList simple_mesh() {
	HittableList objects;
	auto mesh = loadModel("E:/CG_ws/project/myRayTracer/assets/staircase2/models/Mesh001.ply");
	objects.add(mesh);

	return objects;
}

#endif