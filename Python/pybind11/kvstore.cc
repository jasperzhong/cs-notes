#include "kvstore.h"

void KVStore::set(const std::vector<Key>& keys, at::Tensor values) {
  auto keys_size = keys.size();
  for (std::size_t i = 0; i < keys_size; ++i) {
    store_[keys[i]] = values[i];
  }
}

std::vector<at::Tensor> KVStore::get(const std::vector<Key>& keys) {
  auto keys_size = keys.size();
  std::vector<at::Tensor> values(keys_size);
  for (std::size_t i = 0; i < keys_size; ++i) {
    values[i] = store_[keys[i]];
  }

  return values;
}

void AbslKVStore::set(const std::vector<Key>& keys, at::Tensor values) {
  auto keys_size = keys.size();
  for (std::size_t i = 0; i < keys_size; ++i) {
    store_[keys[i]] = values[i];
  }
}

std::vector<at::Tensor> AbslKVStore::get(const std::vector<Key>& keys) {
  auto keys_size = keys.size();
  std::vector<at::Tensor> values(keys_size);
  for (std::size_t i = 0; i < keys_size; ++i) {
    values[i] = store_[keys[i]];
  }

  return values;
}
