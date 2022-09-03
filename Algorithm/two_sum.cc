#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> v{1, 4, 5, 6, 7, 9, 9, 10};
  int target = 12;
  int num = v.size();

  std::sort(v.begin(), v.end());
  int left = 0, right = num - 1;
  bool found = false;
  while (left < right) {
    if ((v[left] + v[right]) == target) {
      found = true;
      break;
    }
    while (left < right && (v[left] + v[right]) > target) right--;
    while (left < right && (v[left] + v[right]) < target) left++;
  }

  if (found) {
    std::cout << v[left] << " " << v[right] << std::endl;
  } else {
    std::cout << "Not found!" << std::endl;
  }
}
