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

include_directories(src/external_library/boost_1_71_0_minimal)
add_subdirectory(src/external_library/boost_1_71_0_minimal/libs/serialization)

add_library(StraightforwardNeuralNetwork ${SOURCES})

target_link_libraries(StraightforwardNeuralNetwork Boost)