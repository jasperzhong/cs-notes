#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

int main()
{
    thrust::device_vector<int> d_vec(10);
    thrust::sequence(d_vec.begin(), d_vec.end());

    auto d_ptr = d_vec.data();
    std::cout << d_ptr << std::endl;
    for (int i = 0; i < d_vec.size(); ++i) {
	std::cout << "*(d_ptr+" << i << ")=" << *(d_ptr + i) << " d_vec[" << i << "]=" << d_vec[i] << std::endl;
    }
}
