#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <typeinfo>

struct TemporalBlock {
  long* target_vertices;
  long* edges;
  float* timestamps;
  long num_edges;
  TemporalBlock* next;
};

void foo() {
  auto mr = rmm::mr::get_current_device_resource();
  std::cout << "Current device resource: " << typeid(*mr).name() << std::endl;

  int num_vertices = 100;
  // create a device vector
  thrust::host_vector<TemporalBlock> blocks(num_vertices);
  thrust::device_vector<TemporalBlock> d_blocks(num_vertices);

  // allocate memory for each block
  for (int i = 0; i < num_vertices; i++) {
    blocks[i].target_vertices =
        (long*)mr->allocate(sizeof(long) * num_vertices);
    blocks[i].edges = (long*)mr->allocate(sizeof(long) * num_vertices);
    blocks[i].timestamps = (float*)mr->allocate(sizeof(float) * num_vertices);
    blocks[i].num_edges = num_vertices;
    blocks[i].next = nullptr;
  }

  // copy the data to the device
  d_blocks = blocks;
}

int main() {
  int size = 1 << 30;  // 1GB

  rmm::mr::cuda_memory_resource mem_res;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_res(
      &mem_res, size, size);
  // 1KiB -> 1MiB
  rmm::mr::set_current_device_resource(&pool_res);
  // rmm::mr::binning_memory_resource<
  //     rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>
  //     bin_res(&pool_res, 10, 20);
  // rmm::mr::set_current_device_resource(&bin_res);
  foo();
}
