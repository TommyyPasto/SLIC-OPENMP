cmake_minimum_required(VERSION 3.10)

project(MidTermAssignment CXX)

find_package(OpenCV REQUIRED)

add_executable(
        slic_app
        main.cpp
        SLIC.cpp
        SLIC.h
)

target_link_libraries(
        slic_app
        ${OpenCV_LIBS}
)

target_include_directories(
        slic_app
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

set_property(TARGET slic_app PROPERTY CXX_STANDARD 20)
set_property(TARGET slic_app PROPERTY CXX_STANDARD_REQUIRED ON)