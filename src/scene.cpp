#include "scene.h"
#include "box.h"

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

	objects.add(make_shared<Box>(point3(130, 0, 65), point3(295, 165, 230), white));
	objects.add(make_shared<Box>(point3(265, 0, 295), point3(430, 330, 460), white));
	return objects;
}