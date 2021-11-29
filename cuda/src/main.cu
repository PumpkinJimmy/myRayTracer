#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cudastart.h>

//__global__ void write_color(float* output) {
//    output[blockIdx.x] = 1.0f;
//    // printf("%d ", blockIdx.x);
//}

__global__ void write_color(float* output, int width, int height) {
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    output[3 * (iy * width + ix)] = float(iy) / (width-1);
    output[3 * (iy * width + ix) + 1] = float(ix)/(height-1);
    output[3 * (iy * width + ix) + 2] = 0.25f;
}

//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);

    const int image_width = 256;
    const int image_height = 256;
    const int cnt = image_width * image_height * 3;
    float* output_h = (float*)malloc(cnt * sizeof(float));
    float* output_d;
    CHECK(cudaMalloc((void**)&output_d, cnt * sizeof(float)));

    dim3 grid(image_width, image_height);
    // dim3 grid(cnt);

    write_color << <grid, 1 >> > (output_d, image_width, image_height);
    // write_color << <grid, 1 >> > (output_d);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(output_h, output_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::ofstream fout("result.ppm");

    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int i = 0; i < cnt; i++) {
        fout << static_cast<int>(255.999 * output_h[i]) << ' ';
    }
    //for (int j = image_height - 1; j >= 0; j--) {
    //    for (int i = 0; i < image_width; i++) {
    //        auto r = double(i) / (image_width - 1);
    //        auto g = double(j) / (image_width - 1);
    //        auto b = 0.25;

    //        int ir = static_cast<int>(255.999 * r);
    //        int ig = static_cast<int>(255.999 * g);
    //        int ib = static_cast<int>(255.999 * b);

    //        fout << ir << ' ' << ig << ' ' << ib << '\n';
    //    }
    //}

    cudaFree(output_d);
    free(output_h);
    cudaDeviceReset();
    return 0;
}