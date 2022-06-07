#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <vector>
#include <iostream>

int main() {
    thrust::device_vector<int> d_vec(10);
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // 1. copy std::vector to thrust::device_vector
    thrust::copy(vec.begin(), vec.end(), d_vec.begin());

    for (int i = 0; i < 10; i++) {
        std::cout << d_vec[i] << std::endl;
    }

    // 2. copy std::vector to thrust::device_ptr
    thrust::device_ptr<int> d_ptr = thrust::device_new<int>(11);
    std::cout << d_ptr.get() << std::endl;
    thrust::copy(vec.begin(), vec.end(), d_ptr.get()+1);
    for (int i = 0; i < 11; i++) {
        std::cout << d_ptr[i] << std::endl;
    }
    thrust::device_delete(d_ptr);

    return 0;
}
