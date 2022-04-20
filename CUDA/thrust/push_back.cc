#include <iostream>
#include <thrust/device_vector.h>

int main() {
	thrust::device_vector<int> d_vec(15);
	std::cout << "size=" << d_vec.size() << " capacity=" << d_vec.capacity() << std::endl;
	d_vec.push_back(1);
	std::cout << "size=" << d_vec.size() << " capacity=" << d_vec.capacity() << std::endl;
}
