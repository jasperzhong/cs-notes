#include <algorithm>
#include <iostream>
#include <vector>

int cnt = 0;

void search(int k, int n, std::vector<int>& permutation,
            std::vector<int>& chosen) {
  if (k == n) {
    std::cout << "permutation #" << cnt << ":\t";
    for (auto x : permutation) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    cnt++;
  } else {
    for (int i = 0; i < n; ++i) {
      if (chosen[i]) continue;
      chosen[i] = 1;
      permutation.push_back(i);
      search(k + 1, n, permutation, chosen);
      chosen[i] = 0;
      permutation.pop_back();
    }
  }
}

void generating_permutations(int k, int n) {
  std::vector<int> permutation(n);
  for (int i = 0; i < n; ++i) {
    permutation[i] = i;
  }

  do {
    std::cout << "permutation #" << cnt << ":\t";
    for (auto x : permutation) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    cnt++;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}

int main() {
  int n = 3;
  long num_permutations = 1;
  for (int i = 1; i <= n; ++i) {
    num_permutations *= i;
  }
  std::cout << "#permutation = " << num_permutations << std::endl;

  std::cout << "method 1:" << std::endl;
  std::vector<int> permutation;
  std::vector<int> chosen(n, 0);
  cnt = 0;
  search(0, n, permutation, chosen);

  std::cout << "method 2:" << std::endl;
  cnt = 0;
  generating_permutations(0, n);
}
