#include "absl/container/flat_hash_map.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::vector<std::string> words = { "hello", "world", "hello", "world" };
    absl::flat_hash_map<std::string, int> counts;
    for (const std::string& word : words) {
        counts[word]++;
    }
    for (const auto& pair : counts) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
}
