# CUDA
## 架构简述
- GPU可以看作是*允许大量进程并发*的协处理器
- CUDA计算本质上是*异构计算*问题：在CUDA上编程必须要直接处理硬件异构问题。
- GPU执行的代码称为*Kernel*
- Execution：*SM, Grid, Block, Warp*
  - SM是**GPU的代码执行单元**
  - Grid: 同一个Kernel并发的所有线程。一个Grid含多个Block。它可以是矩阵式排布的。
  - Block: 同一个SM上运行的线程称为Block。一个Block含多个Thread。它可以是矩阵式排布的。**一个SM同时只能调度一个Block**
  - Warp：中文“线程束”。一个线程束32个线程，**在同一指令周期内Warp的所有线程执行同一个指令**。但由于每个线程有自己的寄存器上下文，因此执行的结果可能是不一样的。线程束分叉会降低性能
- Memory: *Local Memory, Shared Memory, Global Memory*
  - Register: 每一个线程都有自己的寄存器上下文
  - Local Memory: 线程自己的内存
  - Shared Memory: Block内线程共享的内存
  - Global Memory: 全局内存
  - 三种内存类似三级处理器Cache，越往下越大，也越慢。
  - 在核函数中定义的变量默认都是Register变量或者Local Memory
- 编译单文件CUDA程序
  - `nvcc my_cuda_single.cu -o myapp`
  - 注意CUDA是C语言的超集，因此这实际上也是在编译C/C++程序

## 核函数
- 三种声明
  - `__global__`：Host调用的核
  - `__device__`：Device调用的核
  - `__host__`：Host上的通常函数，平常可以省略
    
    注意`__global__`和`__device__`不能同时声明

    有一个显式使用`__host__`的理由：令一个函数同时可以在函数可以在Host或者Device调用。这样可以给出兼容性。
- 核函数的限制
  - **只能访问设备内存**
  - **只能执行设备代码**
  - **__global__不可递归**
  - 返回类型只能是void
  - 不支持C的可变数目参数
  - 不支持静态局部变量
  - 不可以使用STL（但是有替代品Trust）
  - 对非POD类的使用受限（因为内部的指针引用的内存是Host Memory）
  - 向`__global__`传入含虚函数的类
  - **不可使用C标准库**
- 核函数可以
  - 执行任何基本运算和流程控制
  - 调用`__device__`声明的函数（**包括成员函数**）
  - 动态分配内存（`malloc/free`或者`new/delete`）
  - 使用部分C++特性，包括引用、虚函数
  - 创建和使用C++类对象，并利用多态特性，使用虚函数
  - 使用部分模板特性
  - 使用Host的const常量（因为这些常量本质上都可以在编译中优化为硬编码的常量，与跨设备内存问题无关）

## API
- 头文件
  - `#include <cuda_runtime.h>`
- Host API
  - `cudaMalloc(void** p, size_t nBytes)` 分配设备内存
  - `cudaFree()` 回收设备内存
  - 设备和主机之间的内存复制：
    ```
      cudaMemcpy(dst, src, nBytes, flag)
    ``` 

    `flag`取值`cudaMemcpyDeviceToHost`, `cudaMemcpyHostToDevice`

  - `cudaMemcpyToSymbol()`实现**Host上的对象复制到Device上**
    
    注意这里有陷阱：由于Device上不能执行Host代码，因此**带虚方法的对象，在Device Code上无法利用多态**

    这是因为虚函数的实现依赖虚函数表，表中的指针都是指向Host的代码的。

    正确用法是：**只向Device发送数组和POD对象**。如果需要预定义对象，则利用通信的数据**在Device上构造对象**。在Device上构造的对象在Device上使用是可以正常多态的。

  -  `dim3 block(n,n)` 变量，存放Block的Size
  -  `mykernel<<<grid,block>>>(args)` 启动核函数的执行
     
     说明：核函数的执行是非阻塞异步的
  
  - `cudaDeviceSynchronize()`阻塞直到核函数运行完成
  - `cudaDeviceReset()` 重置设备
  - `__global__, __host__, __device__`
  - `cudaError_t` 所有API调用的统一返回类型
  - `cudaGetErrorString(error)` 返回错误信息
  - `__device__`声明用于在Host Code中声明设备上的全局变量。注意：**不可以在Host Code 函数内部这样使用**
  - `__constant__`在Host Code上声明Device上的变量。



- Kernel API
  - `threadIdx.x/threadIdx.y`当前线程在线程“阵列”中的“位置”
  - `blockDim.x/blockDim.y`当前所在Block的Size
  - `blockIdx.x/blockIdx.y`当前Block在Grid的Block“阵列”中的“位置”
  - `__syncthreads()` Block内部线程同步
  - 声明Shared Memory变量。

- 编程实践相关
  - 虚函数/纯虚基类范式使用要小心，**尤其不能使用Host中构造的含虚函数的对象**。因为场景全局信息通常由Host Code处理。但Host到Device的路上只能传数据成员，虚函数表传过去没用了。如果是核函数构造的，那通常也不需要多态（此时更需要反序列化和反射）。
## 调试
讲道理，核函数里面居然可以printf。。。

在项目设置的CUDA -> Device里启用调试信息，然后使用菜单里的拓展->Nsight启动断点调试

比较新的架构用Nsight(Next-gen)调试

## Device动态内存分配
- 直接在Device Code中使用`malloc/free`或者`new/delete`
- 这样分配的内存来自`Global Memory`
- **需要提前开放Device堆空间：`cudaThreadSetLimit(cudaLimitMallocHeapSize, yourSize)`**
- 一般不要再线程中直接new，因为并发的线程很多，这样很可能new一堆。可以考虑运行一些`<<<1,1>>>`的初始化代码
## Shared Memory使用
- 静态
  
  编译时已知，则直接在Device Code中声明`__shared__ int arr[SIZE]`

- 动态
  
  动态分配的Shared Memory如下步骤使用：
  1. Host中指明大小：`mykernel<<<gridDim, blockDim, sharedSize>>>()`
  2. Device中声明：`extern __shared__ int arr[]`；注意**一个kernel中只能声明一次这样的东西，且它的大小就是上一步中指定的**
  3. **(Device)线程之间同步**：`__syncthreads()`

## Vector Types（待完善）

## Thrust（待完善）

## *Advanced（待完善）
### 避免线程束分化

### 提高访问局部性

### 手动展开循环、线程束