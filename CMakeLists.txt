# execute this from build directory : cmake -G"Unix Makefiles" ./.. && make && ./test/Test.out
cmake_minimum_required(VERSION 3.0)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_subdirectory(src)
add_subdirectory(test)