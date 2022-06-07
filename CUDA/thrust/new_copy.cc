#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <iostream>

struct Data {
  int x;
  int y;
};

int main() {
  auto p = thrust::device_new<Data>();
  Data data{1, 2};
  *p = data;
  thrust::device_delete(p);
  return 0;
}
