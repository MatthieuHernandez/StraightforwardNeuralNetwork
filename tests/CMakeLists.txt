include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        03597a01ee50ed33e9dfd640b249b4be3799d395 # v1.13.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

include_directories(../src src)
include_directories(../src/external_library/boost_1_80_0_minimal)
include_directories(../src/external_library/cpp-static-block)

add_library(ExtendedGTest ${SOURCES})

target_link_libraries(ExtendedGTest gtest)

add_subdirectory(audiofile)
add_subdirectory(unit_tests)
add_subdirectory(dataset_tests)
