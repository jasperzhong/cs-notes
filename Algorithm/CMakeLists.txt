cmake_minimum_required(VERSION 3.19)
project(Algorithm)

file(GLOB cuda_learn_sources "*.cc")
foreach(source ${cuda_learn_sources})
    get_filename_component(target ${source} NAME_WLE)
    add_executable(${target} ${source})
endforeach()
