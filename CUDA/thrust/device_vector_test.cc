#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

struct Data {
  int x;
  int y;
};

int main() {
  thrust::device_vector<Data> d_data(10);

  // Data* data = static_cast<thrust::device_ptr<Data>>(d_data.data() + 5).get();
  d_data[5] = {1, 2};

  Data data_on_host = d_data[5];
  std::cout << data_on_host.x << " " << data_on_host.y << std::endl;
}
