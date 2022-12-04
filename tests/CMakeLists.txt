file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_subdirectory(googletest-release-1.10.0)

include_directories(../src src)
include_directories(../src/external_library/boost_1_80_0_minimal)

add_library(ExtendedGTest ${SOURCES})

target_link_libraries(ExtendedGTest gtest)

add_subdirectory(audiofile)
add_subdirectory(unit_tests)
add_subdirectory(dataset_tests)
