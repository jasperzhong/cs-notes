## CUDA 


### Unified Virtual Memory (UVA)

UVA实现的功能就是**demand paging**, 和传统OS的demand paging功能非常类似. 当要访问的page不在当前device上，就会触发page fault，GPU线程暂停，等待migrate完成，然后继续执行.  

Unified Virtual Memory (UVA)是实现以下两个features的底层技术:
- Unified Memory (UM): 使用`cudaMallocManaged()`, one-pointer-for-both
- Mapped Memory (Zero-copy): 分两种
    - one-pointer-for-both: `cudaHostAlloc()`带默认flag. 
    - two-pointers: `cudaHostAlloc()`带`cudaHostAllocMapped`flag 或者是`cudaHostRegister()`带`cudaHostRegisterMapped`flag. 

这两个features有重大区别:
- UM是会自动move data between host and device. 比如从host memory transfer到device memory后，下次再访问同样的数据，就可以直接在device上读取了. 
- Mapped memory的data是留在pinned memory (page-lock), 在需要的时候被transferred到GPU上，但不会留在GPU上（除非GPU代码要求这样）. 如果访问数据多次，transfer也会执行多次. 


可以从下面几个实验看出这两个方法区别:

1. UM

`uva_add_grid.cu`和`uva_add_grid_prefetch.cu`(带prefetch优化)都是用`cudaMallocManaged()`这个API，一个pointer就可以同时在host和device上使用. 执行三次`add()`kernel结果如下:


```sh
==32983== NVPROF is profiling process 32983, command: ./uva_add_grid
==32983== Profiling application: ./uva_add_grid
==32983== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  3.7599ms         3  1.2533ms  32.126us  3.6948ms  add(int, float*, float*)
      API calls:   98.22%  320.97ms         2  160.48ms  17.169us  320.95ms  cudaMallocManaged
                    1.15%  3.7500ms         1  3.7500ms  3.7500ms  3.7500ms  cudaDeviceSynchronize
                    0.44%  1.4223ms         2  711.16us  696.96us  725.36us  cuDeviceTotalMem
                    0.15%  503.70us       194  2.5960us     214ns  107.52us  cuDeviceGetAttribute
                    0.02%  68.307us         3  22.769us  6.1690us  52.629us  cudaLaunchKernel
                    0.02%  51.741us         2  25.870us  22.194us  29.547us  cuDeviceGetName
                    0.01%  18.270us         2  9.1350us  2.4560us  15.814us  cuDeviceGetPCIBusId
                    0.00%  1.3180us         4     329ns     211ns     549ns  cuDeviceGet
                    0.00%  1.2800us         2     640ns     284ns     996ns  cuDeviceGetCount
                    0.00%     864ns         2     432ns     400ns     464ns  cuDeviceGetUuid
```

可以看到，`add()` kernel Min时间非常短，而Max时间很长. 说明第一次执行的时候，需要非常多的migration，所以很慢. 但是后面两次执行，数据已经move到device上，所以速度执行很快. 


```sh
==3960== NVPROF is profiling process 3960, command: ./uva_add_grid_prefetch
==3960== Profiling application: ./uva_add_grid_prefetch
==3960== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  98.269us         3  32.756us  32.223us  33.215us  add(int, float*, float*)
      API calls:   98.82%  331.36ms         2  165.68ms  18.608us  331.34ms  cudaMallocManaged
                    0.44%  1.4754ms         2  737.68us  732.79us  742.57us  cuDeviceTotalMem
                    0.29%  970.27us         1  970.27us  970.27us  970.27us  cudaDeviceSynchronize
                    0.24%  818.66us         2  409.33us  188.77us  629.89us  cudaMemPrefetchAsync
                    0.16%  541.31us       194  2.7900us     221ns  119.67us  cuDeviceGetAttribute
                    0.02%  62.981us         3  20.993us  7.1450us  45.289us  cudaLaunchKernel
                    0.02%  56.275us         2  28.137us  22.974us  33.301us  cuDeviceGetName
                    0.00%  15.372us         2  7.6860us  2.5360us  12.836us  cuDeviceGetPCIBusId
                    0.00%  1.7950us         1  1.7950us  1.7950us  1.7950us  cudaGetDevice
                    0.00%  1.3470us         4     336ns     214ns     648ns  cuDeviceGet
                    0.00%  1.1510us         2     575ns     287ns     864ns  cuDeviceGetCount
                    0.00%     744ns         2     372ns     337ns     407ns  cuDeviceGetUuid
```

使用了prefetch，可以减少第一次执行`add()`时候`的migration，所以连Max时间都很短.


2. Mapped Memory

`uva_add_grid_mapped_memory_one_pointer.cu`和`uva_add_grid_mapped_memory_two_pointers.cu`分别演示了one-pointer-for-both和two-pointers的情况. 


```sh
==24633== NVPROF is profiling process 24633, command: ./uva_add_grid_mapped_memory_one_pointer
==24633== Profiling application: ./uva_add_grid_mapped_memory_one_pointer
==24633== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.4315ms         3  810.50us  780.81us  863.53us  add(int, float*, float*)
      API calls:   98.66%  343.14ms         2  171.57ms  2.2558ms  340.88ms  cudaHostAlloc
                    0.70%  2.4180ms         1  2.4180ms  2.4180ms  2.4180ms  cudaDeviceSynchronize
                    0.44%  1.5407ms         2  770.36us  769.75us  770.96us  cuDeviceTotalMem
                    0.16%  552.91us       194  2.8500us     238ns  123.28us  cuDeviceGetAttribute
                    0.02%  76.671us         3  25.557us  6.9940us  59.161us  cudaLaunchKernel
                    0.02%  64.966us         2  32.483us  25.147us  39.819us  cuDeviceGetName
                    0.01%  19.141us         2  9.5700us  4.0510us  15.090us  cuDeviceGetPCIBusId
                    0.00%  1.9200us         4     480ns     264ns     982ns  cuDeviceGet
                    0.00%  1.8820us         2     941ns     398ns  1.4840us  cuDeviceGetCount
                    0.00%     833ns         2     416ns     384ns     449ns  cuDeviceGetUuid
```

```sh
==1813== NVPROF is profiling process 1813, command: ./uva_add_grid_mapped_memory_two_pointers
==1813== Profiling application: ./uva_add_grid_mapped_memory_two_pointers
==1813== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.3711ms         3  790.35us  762.26us  840.78us  add(int, float*, float*)
      API calls:   98.69%  325.72ms         2  162.86ms  1.9749ms  323.74ms  cudaHostAlloc
                    0.72%  2.3633ms         1  2.3633ms  2.3633ms  2.3633ms  cudaDeviceSynchronize
                    0.40%  1.3297ms         2  664.84us  661.13us  668.54us  cuDeviceTotalMem
                    0.15%  481.62us       194  2.4820us     205ns  102.58us  cuDeviceGetAttribute
                    0.02%  54.512us         3  18.170us  5.3070us  41.538us  cudaLaunchKernel
                    0.01%  48.053us         2  24.026us  20.962us  27.091us  cuDeviceGetName
                    0.01%  16.887us         2  8.4430us  1.8690us  15.018us  cuDeviceGetPCIBusId
                    0.00%  10.987us         2  5.4930us     586ns  10.401us  cudaHostGetDevicePointer
                    0.00%  1.3960us         4     349ns     226ns     693ns  cuDeviceGet
                    0.00%  1.0880us         2     544ns     238ns     850ns  cuDeviceGetCount
                    0.00%     669ns         2     334ns     305ns     364ns  cuDeviceGetUuid
```

可以看到，one-pointer-for-both和two-poniters两个方法在性能上差别不大. 但是执行多次kernel所需要的时间却相近，而且仅略小于UM方法(不带prefetch)第一次的时间. 这就验证了Mapped Memory方法的确是每次都需要transfer, 数据并不会隐式地存放在GPU上.
