file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_library(Bitmap ${SOURCES})

target_link_libraries(Bitmap)