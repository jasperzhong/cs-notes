#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>

using NIDType = uint64_t;
using EIDType = uint64_t;
using TimestampType = float;

#define CUDA_CALL(func)                                              \
  {                                                                  \
    cudaError_t e = (func);                                          \
    if (e != cudaSuccess && e != cudaErrorCudartUnloading)           \
      throw thrust::system_error(e, thrust::cuda_category(), #func); \
  }

struct TemporalBlock {
  NIDType* dst_nodes;
  TimestampType* timestamps;
  EIDType* eids;

  std::size_t size;
  std::size_t capacity;

  TemporalBlock* prev;
  TemporalBlock* next;
};

struct DoublyLinkedList {
  TemporalBlock head;
  TemporalBlock tail;
  std::size_t size;

  __host__ __device__ DoublyLinkedList() : size(0) {
    head.prev = nullptr;
    head.next = &tail;
    tail.prev = &head;
    tail.next = nullptr;
    printf("DoublyLinkedList()\n");
  }
};

__host__ __device__ void InsertBlockToDoublyLinkedList(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  auto& list = node_table[node_id];
  auto head_next = list.head.next;
  list.head.next = block;
  block->prev = &list.head;
  block->next = head_next;
  head_next->prev = block;
  list.size++;
}

__global__ void InsertBlockToDoublyLinkedListKernel(
    DoublyLinkedList* node_table, NIDType node_id, TemporalBlock* block) {
  InsertBlockToDoublyLinkedList(node_table, node_id, block);
}

int main() {
  thrust::device_vector<DoublyLinkedList> node_table(1);

  //  auto block = thrust::device_new<TemporalBlock>(1);
  //
  //  InsertBlockToDoublyLinkedListKernel<<<1, 1>>>(
  //      thrust::raw_pointer_cast(node_table.data()), 0, block.get());
  //
  //
  //
  //  CUDA_CALL(cudaDeviceSynchronize());

  return 0;
}
