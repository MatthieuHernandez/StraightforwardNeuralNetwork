file(GLOB SOURCES
    "tools/*.hpp"
    "tools/*.cpp"
    "data/*.hpp"
    "data/*.cpp"
    "neural_network/*.hpp"
    "neural_network/*.cpp"
    "neural_network/*.tpp"
    "neural_network/layer/*.hpp"
    "neural_network/layer/*.cpp"
	"neural_network/layer/*.tpp"
    "neural_network/layer/perceptron/*.hpp"
    "neural_network/layer/perceptron/*.cpp"
    "neural_network/layer/perceptron/activation_function/*.hpp"
    "neural_network/layer/perceptron/activation_function/*.cpp"
)

include_directories(external_library/boost_1_71_0_minimal)
add_subdirectory(external_library/boost_1_71_0_minimal/libs/serialization)

add_library(StraightforwardNeuralNetwork ${SOURCES})

target_link_libraries(StraightforwardNeuralNetwork Boost)