#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern void TestVector(std::vector<int>& arr);
extern std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> TestReturnMultiple(int x);

PYBIND11_MODULE(example, m)
{
    m.def("test_return_multiple", &TestReturnMultiple);
    m.def("test_vector", &TestVector);
}
