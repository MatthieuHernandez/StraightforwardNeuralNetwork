# execute this from build directory : cmake -G"Unix Makefiles" ./.. 
cmake_minimum_required(VERSION 3.0)

project(StraightforwardNeuralNetwork)

add_definitions(-std=c++17)

file(GLOB SOURCES
    "src/tools/*.hpp"
    "src/tools/*.cpp"
    "src/data/*.hpp"
    "src/data/*.cpp"
    "src/neural_network/*.hpp"
    "src/neural_network/*.cpp"
    "src/neural_network/layer/*.hpp"
    "src/neural_network/layer/*.cpp"
    "src/neural_network/layer/perceptron/*.hpp"
    "src/neural_network/layer/perceptron/*.cpp"
    "src/neural_network/layer/perceptron/activation_function/*.hpp"
    "src/neural_network/layer/perceptron/activation_function/*.cpp"
)

set(Boost_DEBUG 1)
set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)

set(BOOST_ROOT src/external_library/boost_1_71_0)
set(BOOST_INCLUDEDIR src/external_library/boost_1_71_0)
set(BOOST_LIBRARYDIR src/external_library/boost_1_71_0/stage/lib)

find_package(Boost 1.71.0 REQUIRED REQUIRED serialization)
if(NOT Boost_FOUND)
    execute_process(COMMAND git --version)
    message( SEND_ERROR "Copy boost 1.71.0 in src/external_library")
endif()

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_library(StraightforwardNeuralNetwork ${SOURCES})

target_link_libraries(StraightforwardNeuralNetwork ${Boost_LIBRARIES} )