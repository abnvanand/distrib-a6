cmake_minimum_required(VERSION 3.15)
project(distrib_a6)

set(CMAKE_CXX_STANDARD 17)

add_executable(distrib_a6 main.cpp)
include_directories(/home/abnv/.openmpi/include)
