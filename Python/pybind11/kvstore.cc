#include "kvstore.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(kvstore, m) {
  py::class_<KVStore>(m, "KVStore")
      .def(py::init<>())
      .def("set", &KVStore::set)
      .def("get", &KVStore::get)
      .def("memory_usage", &KVStore::memory_usage);
}
