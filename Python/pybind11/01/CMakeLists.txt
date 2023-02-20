cmake_minimum_required(VERSION 3.18)
project(pybind11_example)

add_subdirectory(pybind11)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cc)
pybind11_add_module(example ${SRC_FILES})
