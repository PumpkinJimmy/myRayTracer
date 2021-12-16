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
#include "scene.h"

std::vector<std::vector<color>> gCanvas;		//Canvas
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

// int seed = std::random_device()();
int seed = 0;

color ray_color(const ray& r, const color& background, const Hittable& world, int depth)
{
	hit_record rec;

	if (depth <= 0)
		return color(0, 0, 0);

	// If the ray hits nothing, return the background color.
	if (!world.hit(r, 0.0001, inf, rec))
		return background;
	ray scattered;
	color attenuation;
	color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		return emitted;
	return emitted + attenuation * ray_color(scattered, background, world,
		depth - 1);
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

	switch (8) {
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
		// world = cornell_box();
		world = BVHNode(cornell_box(), 0, 0);
		background = color(0, 0, 0);
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		samples_per_pixel = 500;
		aperture = 0;
		break;
	case 7:
		world = BVHNode(earth(), 0, 0);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 8:
		world = BVHNode(final_scene2(), 0, 0);
		samples_per_pixel = 200;
		background = color(0, 0, 0);
		lookfrom = point3(478, 278, -600);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		aperture = 0;
		break;

	}
	
	// auto world = random_scene();

	

	Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);

	// Render

	// The main ray-tracing based rendering loop
	// TODO: finish your own ray-tracing renderer according to the given tutorials

// #pragma omp parallel for schedule(dynamic)
		//for (int j = image_height - 1; j >= 0; j--)
		//{
		//	for (int i = 0; i < image_width; i++)
		//	{
		//		for (int s = 0; s < samples_per_pixel; s++) {
		//			auto u = (i + random_double()) / (image_width - 1);
		//			auto v = (j + random_double()) / (image_height - 1);
		//			ray r = cam.get_ray(u, v);
		//			render_buf[j][i] += ray_color(r, background, world, max_depth)/ samples_per_pixel;
		//		}
		//		write_color(i, j, render_buf[j][i]);
		//	}
		//}
	
	
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
				/*if (i == 400 && j == 400) {
					auto tmp = render_buf[j][i] / (s + 1);
					printf("%lf %lf %lf\n", tmp.r(), tmp.g(), tmp.b());
				}*/
			}
		}
		sprintf_s(windowTitle, "%d", s);
	}


	double endFrame = clock();
	double timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;
	std::cout << "Ray-tracing based rendering over..." << std::endl;
	std::cout << "The rendering task took " << timeConsuming << " seconds" << std::endl;
}