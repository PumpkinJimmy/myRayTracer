/*The MIT License (MIT)

Copyright (c) 2021-Present, Wencong Yang (yangwc3@mail2.sysu.edu.cn).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>

#include "common.h"


#include "WindowsApp.h"
#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"
#include "aarect.h"
#include "texture.h"

static std::vector<std::vector<color>> gCanvas;		//Canvas
static char windowTitle[100] = "";

// The width and height of the screen
// const auto aspect_ratio = 3.0 / 2.0;
const auto aspect_ratio = 1.0;
const int gWidth = 800;
// const int gWidth = 480;
//const int gHeight = static_cast<int>(gWidth / aspect_ratio);
const int gHeight = 800;

color render_buf[gHeight][gWidth];

void rendering();

color ray_color(const ray& r, const color& background, const Hittable& world, int depth)
{
	hit_record rec;

	if (depth <= 0)
		return color(0, 0, 0);

	// If the ray hits nothing, return the background color.
	if (!world.hit(r, 0.001, inf, rec))
		return background;
	ray scattered;
	color attenuation;
	color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		return emitted;
	return emitted + attenuation * ray_color(scattered, background, world,
		depth - 1);
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
	return objects;
}




int main(int argc, char* args[])
{
	// Create window app handle
	WindowsApp::ptr winApp = WindowsApp::getInstance(gWidth, gHeight, "Ray Tracing");
	if (winApp == nullptr)
	{
		std::cerr << "Error: failed to create a window handler" << std::endl;
		return -1;
	}

	// Memory allocation for canvas
	gCanvas.resize(gHeight, std::vector<color>(gWidth));

	// Launch the rendering thread
	// Note: we run the rendering task in another thread to avoid GUI blocking
	std::thread renderingThread(rendering);

	// Window app loop
	while (!winApp->shouldWindowClose())
	{
		// Process event
		winApp->processEvent();

		// Display to the screen
		winApp->updateScreenSurface(gCanvas);
		winApp->setWindowTitle(windowTitle);
	}

	renderingThread.join();

	return 0;
}

void write_color(int x, int y, color pixel_color)
{
	// Out-of-range detection
	if (x < 0 || x >= gWidth)
	{
		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
		return;
	}

	if (y < 0 || y >= gHeight)
	{
		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
		return;
	}

	// Note: x -> the column number, y -> the row number
	//gCanvas[y][x] = clamp_color(pixel_color);
	gCanvas[y][x] = clamp_color(sqrt(pixel_color));
}

void rendering()
{
	double startFrame = clock();

	printf("final-term Project (built %s at %s) \n", __DATE__, __TIME__);
	std::cout << "Ray-tracing based rendering launched..." << std::endl;

	// Image

	const int image_width = gWidth;
	const int image_height = gHeight;
	
	const int max_depth = 50;

	int samples_per_pixel = 500;
	// World
	// BVHNode world(random_scene(), 0.0001, inf);

	BVHNode world;
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.1;
	color background(color(0.5, 0.7, 1.0));
	double vfov = 20.0;

	switch (6) {
	default:
	case 0:
		world = BVHNode(random_scene(), 0, 0);
		break;
	
	case 5:
		world = BVHNode(simple_light(), 0, 0);
		background = color(0, 0, 0);
		lookfrom = point3(26, 3, 6);
		lookat = point3(0, 2, 0);
		vfov = 20.0;
		break;
	case 6:
		world = BVHNode(cornell_box(), 0, 0);
		background = color(0, 0, 0);
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		samples_per_pixel = 2000;
		break;
	}
	
	// auto world = random_scene();

	

	Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);

	//HittableList world;

	/*point3 lookfrom(3, 3, 2);
	point3 lookat(0, 0, -1);
	vec3 vup(0, 1, 0);
	auto dist_to_focus = (lookfrom - lookat).length();
	auto aperture = 2.0;

	Camera cam(lookfrom,lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);*/

	/*auto material_ground = Lambertian::create(color(0.8, 0.8, 0.0));
	auto material_center = Lambertian::create(color(0.1, 0.2, 0.5));
	auto material_left = Dielectric::create(1.5);
	auto material_right = Metal::create(color(0.8, 0.6, 0.2), 0.0);

	world.add(Sphere::create(point3(0.0, -100.5, -1.0), 100.0, 
		material_ground));
	world.add(Sphere::create(point3(0.0, 0.0, -1.0), 0.5, 
		material_center));
	world.add(Sphere::create(point3(-1.0, 0.0, -1.0), 0.5, 
		material_left));
	world.add(Sphere::create(point3(-1.0, 0.0, -1.0), -0.45,
		material_left));
	world.add(Sphere::create(point3(1.0, 0.0, -1.0), 0.5, 
		material_right));*/
	/*auto R = cos(pi / 4);
	auto material_left = Lambertian::create(color(0, 0, 1));
	auto material_right = Lambertian::create(color(1, 0, 0));

	world.add(Sphere::create(point3(-R, 0, -1), R, material_left));
	world.add(Sphere::create(point3(R, 0, -1), R, material_right));*/

	// Render

	// The main ray-tracing based rendering loop
	// TODO: finish your own ray-tracing renderer according to the given tutorials

	

	for (int s = 0; s < samples_per_pixel; s++) {
#pragma omp parallel for schedule(dynamic)
		for (int j = image_height - 1; j >= 0; j--)
		{
			for (int i = 0; i < image_width; i++)
			{
				auto u = (i + random_double()) / (image_width - 1);
				auto v = (j + random_double()) / (image_height - 1);
				ray r = cam.get_ray(u, v);
				render_buf[j][i] += ray_color(r, background, world, max_depth);
				write_color(i, j, render_buf[j][i]/ (s+1));
			}
		}
		sprintf_s(windowTitle, "%d", s);
	}


	double endFrame = clock();
	double timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;
	std::cout << "Ray-tracing based rendering over..." << std::endl;
	std::cout << "The rendering task took " << timeConsuming << " seconds" << std::endl;
}