#include "kvstore.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

void KVStore::set(const std::vector<Key>& keys, const at::Tensor& values) {
  const std::size_t size = keys.size();
  for (size_t i = 0; i < size; ++i) {
    store_[keys[i]] = values[i];
  }
}

std::vector<at::Tensor> KVStore::get(const std::vector<Key>& keys) {
  const auto size = keys.size();
  std::vector<at::Tensor> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = store_[keys[i]];
  }
  return values;
}

PYBIND11_MODULE(kvstore, m) {
  py::class_<KVStore>(m, "KVStore")
      .def(py::init<>())
      .def("set", &KVStore::set)
      .def("get", &KVStore::get)
      .def("memory_usage", &KVStore::memory_usage);
}
