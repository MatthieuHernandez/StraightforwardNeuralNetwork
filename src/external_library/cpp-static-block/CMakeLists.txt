file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_library(cpp-static-block ${SOURCES})
target_link_libraries(cpp-static-block)
set_target_properties(cpp-static-block PROPERTIES LINKER_LANGUAGE CXX)