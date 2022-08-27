#include <iostream>
#include <vector>

int cnt = 0;

// method 1: backtracking
void search(int k, int n, std::vector<int>& subset) {
  if (k == n) {
    std::cout << "subset #" << cnt << ":\t";
    for (auto x : subset) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    cnt++;
  } else {
    subset.push_back(k);
    search(k + 1, n, subset);
    subset.pop_back();
    search(k + 1, n, subset);
  }
}

// method 2
void generating_subsets(int k, int n) {
  for (int i = 0; i < (1 << n); ++i) {
    std::vector<int> subset;
    for (int j = 0; j < i; j++) {
      if (i & (1 << j)) subset.push_back(j);
    }
    std::cout << "subset #" << cnt << ":\t";
    for (auto x : subset) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    cnt++;
  }
}

int main() {
  int n = 3;
  std::cout << "#subset = " << (1 << n) << std::endl;

  std::cout << "method 1:" << std::endl;
  std::vector<int> subset;
  cnt = 0;
  search(0, n, subset);

  std::cout << "method 2:" << std::endl;
  cnt = 0;
  generating_subsets(0, n);
}
