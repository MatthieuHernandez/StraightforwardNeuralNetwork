# execute this from build directory : cmake -G"Unix Makefiles" ./.. 
cmake_minimum_required(VERSION 3.0)

project(Test)

add_definitions(-std=c++17)

file(GLOB SOURCES
    "unit_tests/*.h"
    "unit_tests/*.cpp"
)

add_executable(Test ${SOURCES})

add_subdirectory(googletest-release-1.10.0)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test gtest StraightforwardNeuralNetwork)


install(TARGETS Test DESTINATION bin)