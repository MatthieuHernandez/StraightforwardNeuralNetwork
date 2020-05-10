file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_subdirectory(googletest-release-1.10.0)

include_directories(../src src)
include_directories(../src/external_library/boost_1_71_0_minimal)

add_library(GTestExtended ${SOURCES})

target_link_libraries(GTestExtended gtest)

add_subdirectory(audiofile)
add_subdirectory(unit_tests)
add_subdirectory(dataset_tests)
