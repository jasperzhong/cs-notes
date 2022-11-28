#ifndef KVSTORE_H
#define KVSTORE_H

#include <torch/extension.h>

#include <unordered_map>
#include <vector>

class KVStore {
 public:
  using Key = unsigned long;
  explicit KVStore(int num_threads = 8) : num_threads_(num_threads) {}
  ~KVStore() = default;

  void set(const std::vector<Key>& keys, const at::Tensor& values) {
    const std::size_t size = keys.size();
    for (size_t i = 0; i < size; ++i) {
      store_[keys[i]] = values[i];
    }
  }

  at::Tensor get(const std::vector<Key>& keys) const {
    // create a buffer to hold the values of the same dtype as the store_
    const auto size = keys.size();
    auto& first_value = store_.at(keys[0]);
    auto values = at::empty({static_cast<long>(size), first_value.sizes()[0]},
                            first_value.options());
#pragma omp parallel for simd num_threads(num_threads_) schedule(static)
    for (size_t i = 0; i < size; ++i) {
      values[i] = store_.at(keys[i]);
    }
    return values;
  }

  std::size_t memory_usage() const {
    // only count the memory usage of the map
    std::size_t total = (sizeof(Key) + sizeof(at::Tensor)) * store_.size();
    return total;
  }

 private:
  std::unordered_map<Key, at::Tensor> store_;
  int num_threads_;
};

#endif  // KVSTORE_H
