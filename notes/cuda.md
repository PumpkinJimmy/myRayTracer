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