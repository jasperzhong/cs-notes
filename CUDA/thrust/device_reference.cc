#include <iostream>
#include <thrust/device_vector.h>
#include <assert.h>

int main() {

	thrust::device_vector<int> v(1, 0);
	thrust::device_reference<int> ref = v[0];

	assert(ref == v[0]);

	assert(&ref == &v[0]);

	v[0] = 13;
	assert(ref == 13);

	thrust::device_ptr<int> ptr = &v[0];
	assert(ref == *ptr);

	assert(&ref == ptr);

	*ptr = 14;
	assert(ref == 14);


	// However, the following code will fail
	// thrust::device_vector<thrust::device_vector<int>> v_of_v(1);
	// thrust::device_reference<thrust::device_vector<int>> v_ref = v_of_v[0];
}
