file(GLOB SOURCES
    "GTestTools.hpp"
)

add_subdirectory(googletest-release-1.10.0)

include_directories(../src src)
include_directories(../src/external_library/boost_1_71_0_minimal)

add_subdirectory(unit_tests)
add_subdirectory(dataset_tests)