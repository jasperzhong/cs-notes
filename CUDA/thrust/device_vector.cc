#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
	thrust::default_random_engine rng(1337);
	thrust::uniform_int_distribution<int> dist;
	thrust::host_vector<int> h_vec(32 << 20);
	thrust::generate(h_vec.begin(), h_vec.end(), [&] {return dist(rng);});

	thrust::device_vector<int> d_vec = h_vec;

	thrust::sort(d_vec.begin(), d_vec.end());
	
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
