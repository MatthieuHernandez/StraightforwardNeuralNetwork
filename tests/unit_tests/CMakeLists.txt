file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "../../examples/*.hpp"
    "../../examples/*.cpp"
)

add_executable(UnitTests.out ${SOURCES})

target_link_libraries(UnitTests.out GTestExtended StraightforwardNeuralNetwork)

install(TARGETS UnitTests.out DESTINATION bin)