#include <tuple>
#include <vector>

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> TestReturnMultiple(int x)
{
    return std::make_tuple(std::vector<int> { x, x + 1, x + 2 },
        std::vector<int> { x + 3, x + 4, x + 5 },
        std::vector<int> { x + 6, x + 7, x + 8 });
}
