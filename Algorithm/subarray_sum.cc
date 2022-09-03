#include <iostream>
#include <vector>

int main() {
  std::vector<int> v{1, 3, 2, 5, 1, 1, 2, 3};
  int target = 8;
  int num = v.size();

  int sum = 0;
  int left = 0, right = 0;
  bool found = false;
  while (right < num) {
    sum += v[right];
    if (sum == target) {
      std::cout << "left: " << left << " right: " << right << std::endl;
      found = true;
      break;
    } else if (sum > target) {
      sum -= v[right];
      sum -= v[left];
      left += 1;
    } else {
      right++;
    }
  }

  if (!found) {
    std::cout << "Not found" << std::endl;
  }
}
