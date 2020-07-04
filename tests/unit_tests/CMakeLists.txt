file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "../../examples/*.hpp"
    "../../examples/*.cpp"
)

add_executable(UnitTests.out ${SOURCES})

target_link_libraries(UnitTests.out ExtendedGTest StraightforwardNeuralNetwork)

install(TARGETS UnitTests.out DESTINATION bin)