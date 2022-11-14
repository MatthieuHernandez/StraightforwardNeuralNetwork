file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_library(Bitmap ${SOURCES})
target_link_libraries(Bitmap)
set_target_properties(Bitmap PROPERTIES LINKER_LANGUAGE CXX)