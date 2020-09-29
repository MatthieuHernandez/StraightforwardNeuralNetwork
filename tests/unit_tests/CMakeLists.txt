file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "../../examples/*.hpp"
    "../../examples/*.cpp"
)

add_executable(UnitTests ${SOURCES})

target_link_libraries(UnitTests ExtendedGTest StraightforwardNeuralNetwork)

install(TARGETS UnitTests DESTINATION bin)