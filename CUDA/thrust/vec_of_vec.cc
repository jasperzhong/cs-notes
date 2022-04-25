#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main()
{
    // thrust does not allow device_vector of device_vector.
    // thrust::device_vector<thrust::device_vector<int>
    // 1. use host_vector of device_vector.
    thrust::host_vector<thrust::device_vector<int>> vec_of_vec(10,
	thrust::device_vector<int>(10));
    for (int i = 0; i < vec_of_vec.size(); ++i) {
	std::cout << vec_of_vec[i].size() << std::endl;
    }

    // 2. use static array
    thrust::device_vector<int> d_vecs[10];
    for (int i = 0; i < vec_of_vec.size(); ++i) {
	std::cout << d_vecs[i].size() << std::endl;
    }

}
