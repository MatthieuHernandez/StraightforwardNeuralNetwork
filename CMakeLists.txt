# execute this from build directory : cmake -G"Unix Makefiles" ./.. && make && ./unit_tests/UnitTest.out
cmake_minimum_required(VERSION 3.0)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:\"libboost_serialization-vc142-mt-x64-1_71.lib\"")
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_subdirectory(src)
add_subdirectory(tests)