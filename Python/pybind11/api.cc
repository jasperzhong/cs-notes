#include <pybind11/pybind11.h>

#include "kvstore.h"
namespace py = pybind11;

PYBIND11_MODULE(kvstore, m) {
  py::class_<KVStore>(m, "KVStore")
      .def(py::init<>())
      .def("set", &KVStore::set)
      .def("get", &KVStore::get)
      .def("memory_usage", &KVStore::memory_usage);

  py::class_<AbslKVStore>(m, "AbslKVStore")
      .def(py::init<>())
      .def("set", &AbslKVStore::set)
      .def("get", &AbslKVStore::get)
      .def("memory_usage", &AbslKVStore::memory_usage);
}
