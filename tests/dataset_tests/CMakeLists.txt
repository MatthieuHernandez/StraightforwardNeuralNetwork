file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "affNIST/*.hpp"
    "affNIST/*.cpp"
    "audio-cats-and-dogs/*.hpp"
    "audio-cats-and-dogs/*.cpp"
    "CIFAR-10/*.hpp"
    "CIFAR-10/*.cpp"
    "daily-min-temperatures/*.hpp"
    "daily-min-temperatures/*.cpp"
    "Fashion-MNIST/*.hpp"
    "Fashion-MNIST/*.cpp"
    "Iris/*.hpp"
    "Iris/*.cpp"
    "MNIST/*.hpp"
    "MNIST/*.cpp"
    "Wine/*.hpp"
    "Wine/*.cpp"
)

include_directories(../audiofile audiofile)
include_directories(.. tests)

add_executable(DatasetTests ${SOURCES})

target_link_libraries(DatasetTests ExtendedGTest AudioFile StraightforwardNeuralNetwork)

set_target_properties(DatasetTests PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "--gtest_catch_exceptions=$<CONFIG:Release>")

install(TARGETS DatasetTests DESTINATION bin)
