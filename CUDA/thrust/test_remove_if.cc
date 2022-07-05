#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <iostream>
#include <random>
#include <vector>

struct is_negative {
  __host__ __device__ bool operator()(const thrust::tuple<int, float>& t) {
    return thrust::get<1>(t) < 0 && thrust::get<0>(t) < 0;
  }
};

int main() {
  std::vector<int> v1;
  std::vector<float> v2;

  std::mt19937 rng;
  std::uniform_int_distribution<int> dist(-100, 100);
  for (int i = 0; i < 100; ++i) {
    int val = dist(rng);
    v1.push_back(val);
    v2.push_back(val);
  }

  thrust::device_vector<int> d_v1(v1.size());
  thrust::copy(v1.begin(), v1.end(), d_v1.begin());

  int* d_v1_ptr_begin = thrust::raw_pointer_cast(&d_v1[0]);
  int* d_v1_ptr_end = thrust::raw_pointer_cast(&d_v1[0]) + d_v1.size();

  thrust::device_vector<float> d_v2(v2.size());
  thrust::copy(v2.begin(), v2.end(), d_v2.begin());

  float* d_v2_ptr_begin = thrust::raw_pointer_cast(&d_v2[0]);
  float* d_v2_ptr_end = thrust::raw_pointer_cast(&d_v2[0]) + d_v2.size();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto new_end =
      thrust::remove_if(thrust::cuda::par.on(stream),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            thrust::device_ptr<int>(d_v1_ptr_begin),
                            thrust::device_ptr<float>(d_v2_ptr_begin))),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            thrust::device_ptr<int>(d_v1_ptr_end),
                            thrust::device_ptr<float>(d_v2_ptr_end))),
                        is_negative());

  int diff = thrust::distance(thrust::make_zip_iterator(thrust::make_tuple(
                                  thrust::device_ptr<int>(d_v1_ptr_begin),
                                  thrust::device_ptr<float>(d_v2_ptr_begin))),
                              new_end);

  thrust::copy(
      thrust::make_zip_iterator(
          thrust::make_tuple(thrust::device_ptr<int>(d_v1_ptr_begin),
                             thrust::device_ptr<float>(d_v2_ptr_begin))),
      new_end,
      thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())));

  std::cout << "v1: ";
  for (int i = 0; i < diff; ++i) {
    std::cout << v1[i] << " ";
  }

  std::cout << std::endl;
  std::cout << "v2: ";
  for (int i = 0; i < diff; ++i) {
    std::cout << v2[i] << " ";
  }

  std::cout << std::endl;

  cudaStreamDestroy(stream);

  return 0;
}
