#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <climits>

namespace py = pybind11;

double test_function(long long x)
{
    // This function computes the sum of the first 2^x integers
    double sum = 0;
    long long max = (1 << x);
    if (max < 0)
    {
        max = std::numeric_limits<long long>::max();
    }
    for (long long i = 0; i < max; i++) {
        sum += i;
    }
    return sum;
}

PYBIND11_MODULE(example, m)
{
    m.def("test_function_release_gil", &test_function, py::call_guard<py::gil_scoped_release>());
    m.def("test_function", &test_function);
}
