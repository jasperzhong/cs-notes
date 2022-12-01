cmake_minimum_required(VERSION 3.18)
project(pybind11_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(third_party/pybind11)
pybind11_add_module(kvstore kvstore.cc api.cc)

add_subdirectory(third_party/abseil-cpp)

target_link_libraries(kvstore PRIVATE absl::flat_hash_map)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(kvstore PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")


