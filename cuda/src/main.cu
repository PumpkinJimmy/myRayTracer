#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <curand_kernel.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "cudastart.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "model.cuh"
#include "common.cuh"
#include "material.cuh"
#include "cutil_math.h"
#include "hittable.cuh"
#define inf 1000000000

//__constant__ SphereData sd[] = {
//    {{0, 0, -1}, 0.5},
//    {{0, -100.5, -1}, 100},
//    {{-1.0, 0.0, -1.0}, 0.5},
//    {{1.0, 0.0, -1.0}, 0.5}
//};
__device__ Hittable** sps;
__device__ int sphereNumber;


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
    // Sphere** spheres = sps;
    //const int sphereNumber = 5;
    auto spheres = sps;
    
    
    Ray scattered = r;
    color radiance = color{ 1,1,1 };
    
    for (int bounce = 0; bounce < depth; bounce++) {
        hit_record rec;
        float tmax = inf;
        for (int i = 0; i < sphereNumber; i++) {
            hit_record tmp;
            if (spheres[i]->hit(scattered, 0.001, tmax, tmp)) {
                rec = tmp;
                tmax = tmp.t;
            }
        }
        color attenuation;
        if (tmax < inf && rec.mat_ptr->scatter(scattered, rec, attenuation, scattered, randState)) {
            radiance *= attenuation;
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

    // ======== Scene 1 ========
    //auto mat = Lambertian::create(make_color(0.5, 0.5, 0.5));
    //auto sphere = Sphere::create(make_point3(0,0,-1), 0.5, mat);
    //auto sphere2 = Sphere::create(make_point3(0,-100.5, -1), 100, mat);
    //sps = new Hittable::Ptr[2];
    //sps[0] = sphere;
    //sps[1] = sphere2;
    //sphereNumber = 2;

    // ======== Scene 2 =========
    //sps = new Hittable::Ptr [5];
    //sphereNumber = 5;
    //if (sps == NULL) {
    //    printf("Mem. Allocate Fail\n");
    //    return;
    //}
    //Material::Ptr mat_ground = new Lambertian(color{ 0.8, 0.8, 0.0 });
    //Material::Ptr mat_center = new Lambertian(color{ 0.1, 0.2, 0.5 });
    //Material::Ptr mat_left = new Dielectric(1.5);
    //Material::Ptr mat_right = new Metal(color{ 0.8, 0.6, 0.2 }, 0.0);

    //Sphere* s0 = new Sphere(point3{ 0.0, -100.5, -1.0 }, 100.0, mat_ground);
    //Sphere* s1 = new Sphere(point3{ 0.0, 0.0, -1.0 }, 0.5, mat_center);
    //Sphere* s2 = new Sphere(point3{ -1.0, 0.0, -1.0 }, 0.5, mat_left);
    //Sphere* s3 = new Sphere(point3{ 1.0, 0.0, -1.0 }, 0.5, mat_right);
    //Sphere* s4 = new Sphere(point3{ -1.0, 0.0, -1.0 }, -0.4, mat_left);

    //sps[0] = s0; sps[1] = s1; sps[2] = s2; sps[3] = s3; sps[4] = s4;

    // ======= Final Scene =========

    int rcnt = 0;
    int ccnt = 0;
    sps = new Hittable::Ptr[rcnt * ccnt + 4];
    if (sps == NULL) {
        printf("Mem. Allocate Fail\n");
        return;
    }

    curandState randState;
    curand_init(0, 0, 0, &randState);

    int p = 0;

    for (int a = -rcnt / 2; a < rcnt/2; a++) {
        for (int b = -ccnt/2 ; b < ccnt/2 ; b++) {
            auto choose_mat = random_real(&randState);
            point3 center= make_point3(a + 0.9 * random_real(&randState), 0.2, b + 0.9 * random_real(&randState));

            if (length(center - make_point3(4, 0.2, 0)) > 0.9) {
                Material::Ptr sphere_material;
                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = random_vec3(&randState) * random_vec3(&randState);
                    sphere_material = Lambertian::create(albedo);
                    sps[p++] = Sphere::create(center, 0.2, sphere_material);
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = random_vec3(&randState, 0.5, 1);
                    auto fuzz = random_real(&randState, 0, 0.5);
                    sphere_material = Metal::create(albedo, fuzz);
                    sps[p++] = Sphere::create(center, 0.2, sphere_material);

                }
                else {
                    // glass
                    sphere_material = Dielectric::create(1.5);
                    sps[p++] = Sphere::create(center, 0.2, sphere_material);
                }
            }
        }
    }

    Material::Ptr material1 = Dielectric::create(1.5);
    sps[p++] = Sphere::create(make_point3(0, 1, 0), 1.0, material1);

    Material::Ptr material2 = Lambertian::create(make_color(0.4, 0.2, 0.1));
    sps[p++] = Sphere::create(make_point3(-4, 1, 0), 1.0, material2);


    Material::Ptr material3 = Metal::create(make_color(0.7, 0.6, 0.5), 0.0);
    sps[p++] = Sphere::create(make_point3(4, 1, 0), 1.0, material3);

    Material::Ptr ground_meterial = Lambertian::create(make_color(0.5, 0.5, 0.5));
    sps[p++] = Sphere::create(make_point3(0, -1000, 0), 1000, ground_meterial);

    sphereNumber = p;
}

__global__ void render(int image_width, int image_height,color* output, int framenumber, uint hashedframenumber) {


    float aspect_ratio = float(image_width) / image_height;
    // Camera camera(make_point3(0, 0, 0), make_point3(0, 0, -1), make_vec3(0, 1, 0), 90, aspect_ratio, 0.1, 1.0f);
    // Camera camera(make_point3(-2, 2, 1), make_point3(0, 0, -1), make_vec3(0, 1, 0), 90, aspect_ratio, 0.1, 1.0f);
    // Camera camera(make_point3(3, 3, 2), make_point3(0, 0, -1), make_vec3(0, 1, 0), 20, aspect_ratio, 2.0f, sqrtf(27));
    Camera camera(
        make_point3(13, 2, 3),
        make_point3(0, 0, 0),
        make_vec3(0, 1, 0),
        20,
        aspect_ratio,
        0.1f,
        10.0);


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

    // const int sampleNumber = 100;
    const int sampleNumber = 1;
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



void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


//主函数
int main(int argc,char** argv)
{
    // OpenGL初始化
    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, 800, 600);


    //设备初始化
    printf("strating...\n");
    initDevice(0);
    // cudaThreadSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

    printf("Initialize Scene...\n");
    initScene << <1, 1 >> > ();
    cudaDeviceSynchronize();
    printf("Initialize Scene...OK\n");

    // Image
    //const double aspect_ratio = 16.0 / 9;
    //const int image_width = 400;
    const double aspect_ratio = 3.0 / 2;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Scene
    

    // Render
    dim3 block(8,8);
    
    dim3 grid(std::ceilf(float(image_width) / block.x), std::ceilf(float(image_height) / block.y));

    color* output_d = NULL;

    CHECK(cudaMalloc(&output_d, image_width * image_height * sizeof(float3)));



    printf("Rendering scene...\n");

    auto st = cpuSecond();

    render << <grid, block >> > (
        image_width,
        image_height,
        output_d,
        0,
        WangHash(0));

    cudaDeviceSynchronize();

    auto ed = cpuSecond();

    printf("Rendering scene...OK\n");
    printf("Time consume: %lf\n", ed - st);


    color* output_h = (color*)malloc(image_width * image_height * sizeof(float3));

    printf("Saving to file...\n");

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

    printf("Saving to file...OK\n");


    cudaFree(output_d);
    free(output_h);
    cudaDeviceReset();

    // OpenGL MainLoop
    GLuint positionsVBO;
    cudaGraphicsResource* positionVBO_CUDA;
    unsigned size = 800 * 600 * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    printf("Done\n");
    return 0;
}