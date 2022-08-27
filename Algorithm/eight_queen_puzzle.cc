#include <algorithm>
#include <iostream>
#include <vector>

int cnt = 0;
constexpr int n = 8;
std::vector<int> column(n, 0);
std::vector<int> diag1(2 * n - 1, 0);
std::vector<int> diag2(2 * n - 1, 0);

void search(int y) {
  if (y == n) {
    cnt++;
  } else {
    for (int x = 0; x < n; ++x) {
      if (column[x] || diag1[x + y] || diag2[x + n - y - 1]) continue;
      column[x] = diag1[x + y] = diag2[x + n - y - 1] = 1;
      search(y + 1);
      column[x] = diag1[x + y] = diag2[x + n - y - 1] = 0;
    }
  }
}

int main() {
  search(0);
  std::cout << cnt << std::endl; // 92
}
