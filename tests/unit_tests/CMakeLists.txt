file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
    "../../examples/*.hpp"
    "../../examples/*.cpp"
)

add_executable(UnitTests ${SOURCES})

target_link_libraries(UnitTests ExtendedGTest StraightforwardNeuralNetwork)

set_target_properties(UnitTests PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "--gtest_catch_exceptions=$<CONFIG:Release>")

install(TARGETS UnitTests DESTINATION bin)