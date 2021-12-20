#ifndef _CAMERA_H
#define _CAMERA_H
#include <cutil_math.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "common.cuh"
class Camera {
public:
	__host__ __device__ Camera(
		point3 lookfrom,
		point3 lookat,
		vec3 vup,
		float vfov,
		float aspect_ratio,
		float aperture,
		float focus_dist) {

		auto theta = degree_to_radius(vfov);
		auto h = tan(theta / 2);
		auto viewport_height = 2.0 * h;
		auto viewport_width = aspect_ratio * viewport_height;

		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner =
			origin - horizontal / 2 - vertical / 2 - focus_dist * w;
		lens_radius = aperture / 2;
	}

	__device__ Ray get_ray(float s, float t, curandState* randState) const {
		vec3 rd = lens_radius * random_in_unit_disk(randState);
		vec3 offset = u * rd.x + v * rd.y;
		// vec3 offset{ 0, 0, 0 };
		return Ray(
			origin + offset,
			lower_left_corner \
			+ s * horizontal \
			+ t * vertical \
			- origin \
			- offset
		);
	}


private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	double lens_radius;
};

#endif