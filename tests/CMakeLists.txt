file(GLOB SOURCES
    "unit_tests/*.h"
    "unit_tests/*.cpp"
)

add_executable(Test.out ${SOURCES})

add_subdirectory(googletest-release-1.10.0)

include_directories(../src src)
include_directories(../src/external_library/boost_1_71_0_minimal)

target_link_libraries(Test.out gtest StraightforwardNeuralNetwork)

install(TARGETS Test.out DESTINATION bin)