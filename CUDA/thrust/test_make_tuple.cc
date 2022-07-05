#include <thrust/tuple.h>

int main() {
  thrust::tuple<int, int, int> t(1, 2, 3);
  thrust::get<0>(t) = 4;
  thrust::get<1>(t) = 5;
  thrust::get<2>(t) = 6;
  return 0;
}
