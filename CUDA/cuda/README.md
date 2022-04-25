## CUDA 



### Unified Virtual Memory (UVA)

和传统OS上的Virtual memory差不多. 需要维护一个页表，然后做address translation. 当访问的page不在当前device上的时候，触发page fault，然后driver自动做migration. (**demaind paging**)

使用的API是`cudaMallocManaged()`. 返回的是一个`*devPtr`指向allocated memory. 但是这个pointer在CPU和所有GPU上都是有效的. 

一个简单的例子(`uva_add_grid.cu`)
```cu
#include <iostream>
#include <cmath>

__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) 
        y[i] += x[i];
}

int main(void) {
    int N = 1 << 20;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) 
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error:" << maxError << std::endl;
}
```

下面是一个profile例子，可以看到kernel执行占了绝大多数时间. 这其实包括了migrate pages的时间. 当kernel遇到一个absent pages的时候，触发page fault, GPU需要暂停线程的执行，等到migrate完成，才能继续执行. 这个例子中，由于data initialization是发生在CPU上，所以在执行kernel的时候，数据都在CPU上，所以产生了很大的migration overhead. 

```sh
==17094== NVPROF is profiling process 17094, command: ./uva_add_grid
Max error:0
==17094== Profiling application: ./uva_add_grid
==17094== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.0430ms         1  4.0430ms  4.0430ms  4.0430ms  add(int, float*, float*)
      API calls:   97.74%  259.49ms         2  129.74ms  13.106us  259.47ms  cudaMallocManaged
                    1.52%  4.0472ms         1  4.0472ms  4.0472ms  4.0472ms  cudaDeviceSynchronize
                    0.50%  1.3386ms         2  669.28us  667.67us  670.90us  cuDeviceTotalMem
                    0.19%  491.59us       194  2.5330us     206ns  106.06us  cuDeviceGetAttribute
                    0.02%  51.202us         2  25.601us  21.558us  29.644us  cuDeviceGetName
                    0.02%  45.379us         1  45.379us  45.379us  45.379us  cudaLaunchKernel
                    0.01%  17.172us         2  8.5860us  2.3730us  14.799us  cuDeviceGetPCIBusId
                    0.00%  1.5660us         2     783ns     297ns  1.2690us  cuDeviceGetCount
                    0.00%  1.3960us         4     349ns     238ns     653ns  cuDeviceGet
                    0.00%     729ns         2     364ns     342ns     387ns  cuDeviceGetUuid
```
简单的减少migration overhead的方法包括: 在GPU上做data initialization (写一个kernel), 多执行几次kernel (摊销掉migration overhead), 和prefetching. 

这里说下prefetching. CUDA提供了`cudaMemPrefetchAsync()`函数. 在刚刚的例子中加上几行:

```cu
// prefetch the data to the GPU
int device = -1;
cudaGetDevice(&device);
cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);
```

`add` kernel直接快了一个数量级. 
```sh
==30012== NVPROF is profiling process 30012, command: ./uva_add_grid_prefetch Max error:0
==30012== Profiling application: ./uva_add_grid_prefetch
==30012== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  33.759us         1  33.759us  33.759us  33.759us  add(int, float*, float*)
      API calls:   97.84%  275.65ms         2  137.82ms  12.985us  275.63ms  cudaMallocManaged
                    1.17%  3.3061ms         1  3.3061ms  3.3061ms  3.3061ms  cudaDeviceSynchronize
                    0.51%  1.4444ms         2  722.19us  703.82us  740.55us  cuDeviceTotalMem
                    0.24%  671.51us         2  335.76us  126.12us  545.39us  cudaMemPrefetchAsync
                    0.19%  527.75us       194  2.7200us     215ns  114.78us  cuDeviceGetAttribute
                    0.02%  51.819us         2  25.909us  22.251us  29.568us  cuDeviceGetName
                    0.01%  37.179us         1  37.179us  37.179us  37.179us  cudaLaunchKernel
                    0.01%  19.736us         2  9.8680us  2.2490us  17.487us  cuDeviceGetPCIBusId
                    0.00%  9.7350us         4  2.4330us     209ns  9.0160us  cuDeviceGet
                    0.00%  1.8910us         1  1.8910us  1.8910us  1.8910us  cudaGetDevice
                    0.00%  1.6830us         2     841ns     276ns  1.4070us  cuDeviceGetCount
                    0.00%     776ns         2     388ns     334ns     442ns  cuDeviceGetUuid
```

Note: 
- profiling的时候需要加上 `nvprof --unified-memory-profiling off ./uva_add_grid` ([nvprof error code 139 but memcheck OK](https://forums.developer.nvidia.com/t/nvprof-error-code-139-but-memcheck-ok/50329/12))

**Useful Links**
1. [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
2. [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e)

