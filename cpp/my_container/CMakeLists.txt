cmake_minimum_required(VERSION 3.19)
project(MyContainer)

file(GLOB my_container_sources "*.cc")
foreach(source ${my_container_sources})
	get_filename_component(target ${source} NAME_WLE)
	add_executable(${target} ${source})
endforeach()
