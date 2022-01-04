# MyRayTracer

计算机图形学期末大作业。

光线追踪渲染器。

## Feature
- 基本特性
  - Primitives
    - Sphere
    - Axis Aligned Rectangle
    - Box
    - Transform: Translate & RotateY
    - Triangle
    - Mesh
  - Texture
    - Solid Color
    - Checker Texture
    - Image Texure
  - Material
    - Lambertian
    - Metal
    - Dielectric
    - Diffuse Light
    - *Plastic
  - Acceleration
    - BVH
    - OpenMP
  - Other
    - Progressive Rendering
    - SDL：标题显示当前采样数
- 扩展特性
  - PBR
    - 菲涅尔效应
    - Plastic材质
    - *蒙特卡洛积分器 & 重要性采样
    - *Cook-Torrance BRDF
  - CUDA Acceleration
    - CUDA渲染加速
    - 支持球体和Lambertian,Metal,Dielectric三种材质
    - Scene Serialization：针对异构计算背景下的外部场景加载
## Windows平台使用

1. `git clone`到本地
2. 在`build`目录下执行如下命令：
   ```
    cmake ..
   ```
3. 在`build`目录下找到`myRayTracer.sln`解决方案，开始使用

## 说明
- 项目自带了SDL 64位，**请使用CMake 64位配置构建项目**
