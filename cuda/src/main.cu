#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <curand_kernel.h>
#include "cudastart.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "model.cuh"
#include "common.cuh"
#include "material.cuh"
#include "cutil_math.h"
#define inf 1000000000

//__constant__ SphereData sd[] = {
//    {{0, 0, -1}, 0.5},
//    {{0, -100.5, -1}, 100},
//    {{-1.0, 0.0, -1.0}, 0.5},
//    {{1.0, 0.0, -1.0}, 0.5}
//};
__device__ Sphere** sps;


__device__ void write_color(color* output, int width, int row, int col, color c) {
    output[row * width + col] = sqrtf(c);
}
__device__ color ray_color(const Ray& r, int depth, curandState* randState) {


    // ======== Scene 1 ========
    //Lambertian mat({ 0.5, 0.5, 0.5 });
    //Sphere sphere(sd[0].cen, sd[0].r, &mat);
    //Sphere sphere2(sd[1].cen, sd[1].r, &mat);
    //Sphere spheres[] = {
    //    sphere,
    //    sphere2
    //};

    // ======== Scene 2 ========
    /*auto mat_ground = Lambertian(color{ 0.8, 0.8, 0.0 });
    auto mat_center = Lambertian(color{ 0.1, 0.2, 0.5 });
    auto mat_left = Dielectric(1.5);
    auto mat_right = Metal(color{ 0.8, 0.6, 0.2 }, 0.0);

    Sphere s0(point3{ 0.0, -100.5, -1.0 }, 100.0, &mat_ground);
    Sphere s1(point3{ 0.0, 0.0, -1.0 }, 0.5, &mat_center);
    Sphere s2(point3{ -1.0, 0.0, -1.0 }, 0.5, &mat_left);
    Sphere s3(point3{ 1.0, 0.0, -1.0 }, 0.5, &mat_right);
    Sphere s4(point3{ -1.0, 0.0, -1.0 }, -0.4, &mat_left);

    Sphere spheres[] = {
        s0, s1, s2, s3, s4
    };*/
    Sphere** spheres = sps;
    //const int sphereNumber = 5;
    int sphereNumber = 5;
    
    Ray scattered = r;
    color radiance = color{ 1,1,1 };
    for (int bounce = 0; bounce < depth; bounce++) {
        hit_record rec;
        rec.t = inf;
        for (int i = 0; i < sphereNumber; i++) {
            hit_record tmp;
            if (spheres[i]->hit(scattered, 0.0001, inf, tmp)) {
                if (tmp.t < rec.t) {
                    rec = tmp;
                }
                // point3 hitp = r.at(rec.t);
                // vec3 normal = 0.5 + unit_vector(hitp - spheres[i].center) * 0.5;

                // return normal;
                // return color{ 1,0,0 };
            }
        }
        color attenuation;
        if (rec.t < inf && rec.mat_ptr->scatter(scattered, rec, attenuation, scattered, randState)) {
            radiance *= attenuation;
            // return attenuation * ray_color(scattered, depth - 1, randState);
        }
        else {
            vec3 unit_direction = unit_vector(scattered.direction);
            float t = 0.5 * (unit_direction.y + 1.0);
            return radiance * lerp(color{ 1, 1, 1 }, color{ 0.5f, 0.7f, 1.0f }, t);
        }
    }
    return color{ 0, 0, 0 };
}

__global__ void initScene() {
    sps = new Sphere * [5];
    if (sps == NULL) {
        printf("New Fail\n");
    }
    else  printf("New OK\n");

    Material::Ptr mat_ground = new Lambertian(color{ 0.8, 0.8, 0.0 });
    Material::Ptr mat_center = new Lambertian(color{ 0.1, 0.2, 0.5 });
    Material::Ptr mat_left = new Dielectric(1.5);
    Material::Ptr mat_right = new Metal(color{ 0.8, 0.6, 0.2 }, 0.0);

    Sphere* s0 = new Sphere(point3{ 0.0, -100.5, -1.0 }, 100.0, mat_ground);
    Sphere* s1 = new Sphere(point3{ 0.0, 0.0, -1.0 }, 0.5, mat_center);
    Sphere* s2 = new Sphere(point3{ -1.0, 0.0, -1.0 }, 0.5, mat_left);
    Sphere* s3 = new Sphere(point3{ 1.0, 0.0, -1.0 }, 0.5, mat_right);
    Sphere* s4 = new Sphere(point3{ -1.0, 0.0, -1.0 }, -0.4, mat_left);

    sps[0] = s0; sps[1] = s1; sps[2] = s2; sps[3] = s3; sps[4] = s4;
}

__global__ void render(int image_width, int image_height,color* output, int framenumber, uint hashedframenumber) {


    float aspect_ratio = float(image_width) / image_height;
    Camera camera(make_point3(0, 0, 0), make_point3(0, 0, -1), make_vec3(0, 1, 0), 90, aspect_ratio, 0.1, 1.0f);
    // Camera camera(make_point3(-2, 2, 1), make_point3(0, 0, -1), make_vec3(0, 1, 0), 90, aspect_ratio, 0.1, 1.0f);
    // Camera camera(make_point3(3, 3, 2), make_point3(0, 0, -1), make_vec3(0, 1, 0), 20, aspect_ratio, 2.0f, sqrtf(27));

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    // check boundary
    if (x >= image_width || y >= image_height) return;

    // init random
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    curandState randState;
    curand_init(hashedframenumber + threadId, 0, 0, &randState);

    // construct scene
    // Sphere sphere(sd[0]);
    
    // use camera
    // radiance

    const int sampleNumber = 100;
    color accumColor{ 0, 0, 0 };
    for (int i = 0; i < sampleNumber; i++) {
        float u = (x + random_real(&randState)) / (image_width - 1);
        float v = (y + random_real(&randState)) / (image_height - 1);
        Ray ray = camera.get_ray(u, v, &randState);
        // Ray ray = camera.get_ray(u, v);

        color c = ray_color(ray, 50, &randState);
        accumColor += c / sampleNumber;
    }
    write_color(output, image_width, image_height - y - 1, x, accumColor);
    
}





//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);
    // cudaThreadSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    initScene << <1, 1 >> > ();
    

    // Image
     const double aspect_ratio = 16.0 / 9;
     const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Scene
    

    // Render
    dim3 block(8,8);
    
    dim3 grid(std::ceilf(float(image_width) / block.x), std::ceilf(float(image_height) / block.y));

    color* output_d = NULL;

    CHECK(cudaMalloc(&output_d, image_width * image_height * sizeof(float3)));

    render << <grid, block >> > (
        image_width,
        image_height,
        output_d,
        0,
        WangHash(0));

    color* output_h = (color*)malloc(image_width * image_height * sizeof(float3));

    cudaDeviceSynchronize();

    CHECK(cudaMemcpy(output_h, output_d, image_width * image_height * sizeof(float3), cudaMemcpyDeviceToHost));

    FILE* fp = fopen("result.ppm", "w");
    fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);
    for (int i = 0; i < image_width * image_height; i++) {
        fprintf(fp, "%d %d %d ",
            quantize(output_h[i].x),
            quantize(output_h[i].y),
            quantize(output_h[i].z));
    }
    fclose(fp);

    cudaFree(output_d);
    free(output_h);
    cudaDeviceReset();
    return 0;
}