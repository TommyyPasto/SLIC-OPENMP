cmake_minimum_required(VERSION 3.20)
project(slic_openmp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER g++)

find_package(OpenMP REQUIRED)

add_executable(slic_openmp src/main.cpp)

target_link_libraries(slic_openmp PRIVATE OpenMP::OpenMP_CXX)