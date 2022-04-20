cmake_minimum_required(VERSION 3.19)
project(THRUST_LEARN)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

add_executable(hello_thrust hello_thrust.cc)

target_link_libraries(hello_thrust Thrust)
