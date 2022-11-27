#include <iostream>

#include "my_vector.h"

using namespace my_std;

struct Point {
  int x;
  int y;

  Point() = default;
  Point(int x, int y) : x(x), y(y) {}
};

int main() {
  MyVector<int> v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  v1.push_back(4);
  v1.push_back(5);

  MyVector<int> v2 = std::move(v1);

  std::cout << "v1: ";
  for (std::size_t i = 0; i < v1.size(); ++i) {
    std::cout << v1[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "v2: ";
  for (std::size_t i = 0; i < v2.size(); ++i) {
    std::cout << v2[i] << " ";
  }
  std::cout << std::endl;

  MyVector<Point> v3(10);
  for (std::size_t i = 0; i < v3.size(); ++i) {
    v3[i].x = i;
    v3[i].y = i;
  }
  v3.emplace_back(1, 2);
}
