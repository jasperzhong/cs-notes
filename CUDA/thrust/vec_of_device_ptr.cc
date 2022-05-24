#include <iostream>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

int main()
{
    thrust::device_vector<thrust::device_ptr<int>> dev_ptr_vec(10);
    for (int i = 0; i < dev_ptr_vec.size(); ++i) {
        dev_ptr_vec[i] = thrust::device_malloc<int>(10);
    }

    // for (int i = 0; i < dev_ptr_vec.size(); ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         static_cast<thrust::device_ptr<int>&&>(dev_ptr_vec[i])[j] = i * j;
    //     }
    // }

    for (int i = 0; i < dev_ptr_vec.size(); ++i) {
        // need to create a temporary device_ptr
        // since dev_ptr_vec[i] is a device reference
        // to the device_ptr and device_free does not
        // recognize device reference...
        // Unfortunately, device_reference<T> cannot expose members of T, but it can convert to T.
        // https://stackoverflow.com/questions/6624049/passing-thrustdevice-vector-to-a-function-by-reference
        // Solution 1:
        dev_ptr_vec[i];
        // thrust::device_free(ptr);
        // Solution 2:
        // thrust::device_free(static_cast<thrust::device_ptr<int>&&>(dev_ptr_vec[i]));
    }
}
