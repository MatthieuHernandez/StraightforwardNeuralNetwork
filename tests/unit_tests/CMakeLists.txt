file(GLOB SOURCES
    "*.h"
    "*.cpp"
)

add_executable(UnitTests.out ${SOURCES})

target_link_libraries(UnitTests.out gtest StraightforwardNeuralNetwork)

install(TARGETS UnitTests.out DESTINATION bin)