file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "CIFAR-10/*.hpp"
    "CIFAR-10/*.cpp"
    "Iris/*.hpp"
    "Iris/*.cpp"
    "MNIST/*.hpp"
    "MNIST/*.cpp"
    "Wine/*.hpp"
    "Wine/*.cpp"
)

add_executable(DatasetTests.out ${SOURCES})

target_link_libraries(DatasetTests.out gtest StraightforwardNeuralNetwork)

install(TARGETS DatasetTests.out DESTINATION bin)