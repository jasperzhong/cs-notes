#ifndef KVSTORE_H
#define KVSTORE_H

#include <torch/extension.h>

#include <unordered_map>
#include <vector>

class KVStore {
 public:
  using Key = unsigned int;
  explicit KVStore(int num_threads = 8) : num_threads_(num_threads) {}
  ~KVStore() = default;

  void set(const std::vector<Key>& keys, const at::Tensor& values);

  at::Tensor get(const std::vector<Key>& keys);

  std::size_t memory_usage() const {
    // only count the memory usage of the map
    std::size_t total = (sizeof(Key) + sizeof(at::Tensor)) * store_.size();
    return total;
  }

 private:
  std::unordered_map<Key, at::Tensor> store_;
  int num_threads_;
};

void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  const std::size_t size = keys.size();
  for (size_t i = 0; i < size; ++i) {
    store_[keys[i]] = values[i];
  }
}

at::Tensor KVStore::get(const std::vector<Key>& keys) {
  const auto size = keys.size();
  std::vector<at::Tensor> values(size);
#pragma omp parallel for num_threads(num_threads_) schedule(static)
  for (size_t i = 0; i < size; ++i) {
    values[i] = store_[keys[i]];
  }
  return at::stack(values);
}

#endif  // KVSTORE_H
