project(Boost)

file(GLOB SOURCES
    "../../boost/*.h"
    "../../boost/*.hpp"
    "src/*.cpp"
)

if(MSVC)
    string(REPLACE "/sdl" "/sdl-" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

include_directories(../..)

add_library(Boost ${SOURCES})

install(TARGETS Boost DESTINATION bin)

set_target_properties(Boost PROPERTIES CXX_STANDARD 17)