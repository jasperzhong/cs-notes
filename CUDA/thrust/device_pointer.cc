#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

#include <iostream>
#include <map>

struct A {
  int x;
  int y;
  __host__ __device__ int add() const { return x + y; }
};

__global__ void kernel(A* a, int* ans) { *ans = a->add(); }

int main() {
  auto ptr = thrust::device_new<A>(1);
  A a{1, 2};
  *ptr = a;

  auto ans = thrust::device_new<int>(1);
  kernel<<<1, 1>>>(ptr.get(), ans.get());

  std::cout << *ans << std::endl;

  A a2 = *ptr;
  std::cout << a2.add() << std::endl;

  thrust::device_delete(ptr);
}
