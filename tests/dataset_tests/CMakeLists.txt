file(GLOB SOURCES
    "*.h"
    "*.cpp"
    "CIFAR-10/*.h"
    "CIFAR-10/*.cpp"
    "Iris/*.h"
    "Iris/*.cpp"
    "MNIST/*.h"
    "MNIST/*.cpp"
    "Wine/*.h"
    "Wine/*.cpp"
)

add_executable(DatasetTests.out ${SOURCES})

target_link_libraries(DatasetTests.out gtest StraightforwardNeuralNetwork)

install(TARGETS DatasetTests.out DESTINATION bin)