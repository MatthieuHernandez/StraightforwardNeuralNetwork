file(GLOB SOURCES
    "*.hpp"
    "*.cpp"
)

add_library(AudioFile ${SOURCES})

target_link_libraries(AudioFile)