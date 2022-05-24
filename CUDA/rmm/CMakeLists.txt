cmake_minimum_required(VERSION 3.18)
project(RMM_LEARN)
enable_language(CUDA)

find_package(rmm)

set(CMAKE_CUDA_ARCHITECTURES "60")
include_directories($ENV{CONDA_PREFIX}/include)

file(GLOB thrust_learn_sources "*.cc")
foreach(source ${thrust_learn_sources})
	get_filename_component(target ${source} NAME_WLE)
	add_executable(${target} ${source})
        # set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
        target_link_libraries(${target} PUBLIC rmm::rmm)
endforeach()
