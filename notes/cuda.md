# CUDA
## 架构简述
- GPU可以看作是*允许大量进程并发*的协处理器
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
  - **不可递归**
  - 返回类型只能是void
  - 不支持C的可变数目参数
  - 不支持静态局部变量

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

  -  `dim3 block(n,n)` 变量，存放Block的Size
  -  `mykernel<<<grid,block>>>(args)` 启动核函数的执行
     
     说明：核函数的执行是非阻塞异步的
  
  - `cudaDeviceSynchronize()`阻塞直到核函数运行完成
  - `cudaDeviceReset()` 重置设备
  - `__global__, __host__, __device__`
  - `cudaError_t` 所有API调用的统一返回类型
  - `cudaGetErrorString(error)` 返回错误信息
- Kernel API
  - `threadIdx.x/threadIdx.y`当前线程在线程“阵列”中的“位置”
  - `blockDim.x/blockDim.y`当前所在Block的Size
  - `blockIdx.x/blockIdx.y`当前Block在Grid的Block“阵列”中的“位置”
  - `__syncthreads()` Block内部线程同步

## 调试（待完善）
讲道理，核函数里面居然可以printf。。。

## Vector Types（待完善）

## Thrust（待完善）

## *Advanced（待完善）
### 避免线程束分化

### 提高访问局部性

### 手动展开循环、线程束