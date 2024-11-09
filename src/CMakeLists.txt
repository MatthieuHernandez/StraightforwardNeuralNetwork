file(GLOB SOURCES
    "tools/*.cpp"
    "data/*.cpp"
    "neural_network/*.cpp"
    "neural_network/optimizer/*.cpp"
    "neural_network/layer/*.cpp"
    "neural_network/layer/neuron/*.cpp"
    "neural_network/layer/neuron/activation_function/*.cpp"
)


include_directories(external_library/boost_1_80_0_minimal)
add_subdirectory(external_library/boost_1_80_0_minimal/libs/serialization)

include_directories(external_library/bitmap)
add_subdirectory(external_library/bitmap)

include_directories(external_library/cpp-static-block)
add_subdirectory(external_library/cpp-static-block)

add_library(StraightforwardNeuralNetwork ${SOURCES})

target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/data")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network/layer")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network/layer/neuron")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network/layer/neuron/activation_function")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/neural_network/optimizer")
target_include_directories(StraightforwardNeuralNetwork PRIVATE "../include/snn/tools")
target_include_directories(StraightforwardNeuralNetwork INTERFACE "include")

target_link_libraries(StraightforwardNeuralNetwork Boost Bitmap cpp-static-block)
