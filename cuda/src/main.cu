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

__constant__ SphereData sd[] = {
    {{0, 0, -1}, 0.5},
    {{0, -100.5, -1}, 100}
};

__device__ void write_color(color* output, int width, int row, int col, color c) {
    output[row * width + col] = c;
}
__device__ color ray_color(const Ray& r) {
    Sphere sphere(sd[0]);
    Sphere sphere2(sd[1]);

    hit_record rec;
    if (sphere.hit(r, 0.0001, 100000, rec)) {
        point3 hitp = r.at(rec.t);
        vec3 normal = 0.5 + unit_vector(hitp - sphere.center) * 0.5;
        return normal;
        // return color{ 1,0,0 };
    }
    if (sphere2.hit(r, 0.0001, 100000, rec)) {
        point3 hitp = r.at(rec.t);
        vec3 normal = 0.5 + unit_vector(hitp - sphere2.center) * 0.5;
        return normal;
        // return color{ 1,0,0 };
    }
    vec3 unit_direction = unit_vector(r.direction);
    float t = 0.5 * (unit_direction.y + 1.0);
    return lerp(color{ 1, 1, 1 }, color{ 0.5f, 0.7f, 1.0f }, t);
}

__global__ void render(int image_width, int image_height,color* output, int framenumber, uint hashedframenumber) {


    float aspect_ratio = float(image_width) / image_height;
    Camera camera(make_point3(0, 0, 0), make_point3(0, 0, -1), make_vec3(0, 1, 0), 90, aspect_ratio, 0.1, 1.0f);

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    // check boundary
    if (x >= image_width || y >= image_height) return;

    // init random
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    curandState randState;
    curand_init(hashedframenumber + threadId, 0, 0, &randState);

    // construct scene
    Sphere sphere(sd[0]);
    
    // use camera
    // radiance

    const int sampleNumber = 100;
    color accumColor{ 0, 0, 0 };
    for (int i = 0; i < sampleNumber; i++) {
        float u = (x + random_real(&randState)) / (image_width - 1);
        float v = (y + random_real(&randState)) / (image_height - 1);
        Ray ray = camera.get_ray(u, v);

        color c = ray_color(ray);
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

    // scudaDeviceSynchronize();

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