project(Boost)

file(GLOB SOURCES
    "../../boost/*.h"
    "../../boost/*.hpp"
    "src/*.cpp"
)

include_directories(../..)

add_library(Boost ${SOURCES})

install(TARGETS Boost DESTINATION bin)

set_target_properties(Boost PROPERTIES CXX_STANDARD 20)
