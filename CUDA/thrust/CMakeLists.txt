cmake_minimum_required(VERSION 3.19)
project(THRUST_LEARN)
enable_language(CUDA)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

include_directories(include/)

file(GLOB thrust_learn_sources "*.cc")
foreach(source ${thrust_learn_sources})
	get_filename_component(target ${source} NAME_WLE)
	add_executable(${target} ${source})
	set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
	target_link_libraries(${target} Thrust)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_60,code=sm_60>")
endforeach()
