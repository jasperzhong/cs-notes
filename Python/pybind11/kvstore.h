#ifndef KVSTORE_H
#define KVSTORE_H
#include <sys/resource.h>
#include <torch/extension.h>

#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"

class KVStore {
 public:
  using Key = unsigned int;
  using Value = at::Tensor;
  KVStore() = default;
  ~KVStore() = default;

  void set(const std::vector<Key>& keys, at::Tensor values);

  std::vector<at::Tensor> get(const std::vector<Key>& keys);

  float memory_usage() const {
    // get process's CPU memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
  }

 private:
  std::unordered_map<Key, at::Tensor> store_;
};

class AbslKVStore {
 public:
  using Key = unsigned int;
  using Value = at::Tensor;
  AbslKVStore() = default;
  ~AbslKVStore() = default;

  void set(const std::vector<Key>& keys, at::Tensor values);

  std::vector<at::Tensor> get(const std::vector<Key>& keys);

  float memory_usage() const {
    // get process's CPU memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
  }

 private:
  absl::flat_hash_map<Key, at::Tensor> store_;
};

#endif  // KVSTORE_H
