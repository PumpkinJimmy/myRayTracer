#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "cudastart.h"
#include "vec3.cuh"
#include "ray.cuh"

__device__ void write_color(color* output, int width, int row, int col, color c) {
    output[row * width + col] = c;
}
__device__ color ray_color(const Ray& r) {
    vec3 unit_direction = unit_vector(r.direction);
    float t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * make_float3(1., 1., 1.) + t * make_float3(0.5, 0.7, 1.0);
}

__global__ void render(int image_width, int image_height,color* output) {
    float aspect_ratio = float(image_width) / image_height;
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;
    point3 origin = make_point3(0, 0, 0);
    vec3 horizontal = make_vec3(viewport_width, 0, 0);
    vec3 vertical = make_vec3(0, viewport_height, 0);
    point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_vec3(0, 0, focal_length);


    int row = blockIdx.y;
    int col = blockIdx.x;
    float u = float(row) / (image_width - 1);
    float v = float(col) / (image_height - 1);
    Ray ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    point3 target = lower_left_corner + u * horizontal + v * vertical;
    color c = ray_color(ray);
    write_color(output, image_width, row, col, c);
    // printf("(%d %d) %f <%f,%f,%f> <%f, %f>\n", row, col, ray.direction.y, target.x, target.y, target.z, u, v);
}

int quantize(float f) {
    return static_cast<int>(f * 255.999);
}

//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);

    // Image
    // const double aspect_ratio = 16.0 / 9;
    // const int image_width = 400;
    const double aspect_ratio = 1.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;
    point3 origin = make_point3(0, 0, 0);
    vec3 horizontal = make_vec3(viewport_width, 0, 0);
    vec3 vertical = make_vec3(0, viewport_height, 0);
    point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_vec3(0, 0, focal_length);

    // Render
    dim3 block(1,1);
    dim3 grid(image_width / block.x, image_height / block.y);

    color* output_d = NULL;
    CHECK(cudaMalloc(&output_d, image_width * image_height * sizeof(float3)));

    render << <grid, block >> > (
        image_width,
        image_height,
        output_d);

    color* output_h = (color*)malloc(image_width * image_height * sizeof(float3));

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


    /*const int cnt = image_width * image_height * 3;
    float* output_h = (float*)malloc(cnt * sizeof(float));
    float* output_d;
    CHECK(cudaMalloc((void**)&output_d, cnt * sizeof(float)));

    dim3 grid(image_width, image_height);*/
    // dim3 grid(cnt);

    //write_color << <grid, 1 >> > (output_d, image_width, image_height);
    //// write_color << <grid, 1 >> > (output_d);

    //CHECK(cudaDeviceSynchronize());
    //CHECK(cudaMemcpy(output_h, output_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
    //
    //std::ofstream fout("result.ppm");

    //fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    //for (int i = 0; i < cnt; i++) {
    //    fout << static_cast<int>(255.999 * output_h[i]) << ' ';
    //}
    ////for (int j = image_height - 1; j >= 0; j--) {
    ////    for (int i = 0; i < image_width; i++) {
    ////        auto r = double(i) / (image_width - 1);
    ////        auto g = double(j) / (image_width - 1);
    ////        auto b = 0.25;

    ////        int ir = static_cast<int>(255.999 * r);
    ////        int ig = static_cast<int>(255.999 * g);
    ////        int ib = static_cast<int>(255.999 * b);

    ////        fout << ir << ' ' << ig << ' ' << ib << '\n';
    ////    }
    ////}

    //cudaFree(output_d);
    //free(output_h);
    cudaFree(output_d);
    free(output_h);
    cudaDeviceReset();
    return 0;
}