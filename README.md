# MyRayTracer

计算机图形学期末大作业。

光线追踪渲染器。

## Feature
- Model
  - Sphere
  - Axis Aligned Rectangle
  - Box
  - Transform: Translate & RotateY
- Texture
  - Solid Color
  - Diffuse Light
  - Checker Texture
  - Image Texure
 - Material
  - Lambertian
  - Metal
  - Dielectric
 - BVH
 - CUDA Acceleration
   - by now, only support sphere & simple material, no texture

## Windows平台使用

1. `git clone`到本地
2. 在`build`目录下执行如下命令：
   ```
    cmake ..
   ```
3. 在`build`目录下找到`myRayTracer.sln`解决方案，开始使用

## 说明
- 项目自带了SDL 64位，**请使用CMake 64位配置构建项目**
