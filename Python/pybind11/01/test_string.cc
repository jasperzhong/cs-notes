#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(test_string, m)
{
    m.def("test_string", [](const std::string& s) {
	std::vector<uint8_t> vec(s.begin(), s.end());
	for (auto ele : vec) {
	    std::cout << static_cast<int>(ele);
	}
	std::cout << std::endl;
    });
}
