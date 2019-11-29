# execute this from build directory : cmake -G"Unix Makefiles" ./.. 
cmake_minimum_required(VERSION 3.0)

project(Boost)

add_definitions(-std=c++17)

file(GLOB SOURCES
    "../../boost/*.h"
    "../../boost/*.hpp"
    "src/*.cpp"
)

include_directories(../..)

add_library(Boost ${SOURCES})

install(TARGETS Boost DESTINATION bin)