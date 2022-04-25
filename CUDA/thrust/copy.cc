#include <iostream>
#include <thrust/device_vector.h>

int main() {
	thrust::device_vector<int> v1(1, 3);
	thrust::device_vector<int> v2(1);
	v2 = v1;

	std::cout << "v1=" << v1[0] << " v2=" << v2[0] << std::endl;
	
	v2[0] = 4;
	std::cout << "v1=" << v1[0] << " v2=" << v2[0] << std::endl;
}
