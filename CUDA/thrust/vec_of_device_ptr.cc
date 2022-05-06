#include <iostream>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

__global__ void add_by_one(thrust::device_ptr<int>* arrays, int num_arrays, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_arrays) {
        for (int i = 0; i < num_elements; i++) {
            arrays[idx][i] += 1;
        }
    }
}

int main()
{
    thrust::device_vector<thrust::device_ptr<int>> dev_ptr_vec(10);
    for (int i = 0; i < dev_ptr_vec.size(); ++i) {
        dev_ptr_vec[i] = thrust::device_malloc<int>(10);
    }

    auto dev_ptr_vec_begin = thrust::raw_pointer_cast(dev_ptr_vec.data());
    int threadsPerBlock = 256;
    int blocksPerGrid = (10 + threadsPerBlock - 1) / threadsPerBlock;
    add_by_one<<<blocksPerGrid, threadsPerBlock>>>(dev_ptr_vec_begin, 10, 10);

    for (int i = 0; i < dev_ptr_vec.size(); ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << static_cast<thrust::device_ptr<int>>(dev_ptr_vec[i])[j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < dev_ptr_vec.size(); ++i) {
        // need to create a temporary device_ptr
        // since dev_ptr_vec[i] is a device reference
        // to the device_ptr and device_free does not
        // recognize device reference...
        // Unfortunately, device_reference<T> cannot expose members of T, but it can convert to T.
        // https://stackoverflow.com/questions/6624049/passing-thrustdevice-vector-to-a-function-by-reference
        // Solution 1:
        // thrust::device_ptr<int> ptr(dev_ptr_vec[i]);
        // thrust::device_free(ptr);
        // Solution 2:
        thrust::device_free(static_cast<thrust::device_ptr<int>>(dev_ptr_vec[i]));

        // However, both solutions need to copy the device_ptr to the host (memcpy triggered).
    }
}
