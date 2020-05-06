file(GLOB SOURCES
    "*.h"
)

add_library(AudioFile ${SOURCES})

target_link_libraries(AudioFile)

set_target_properties(AudioFile PROPERTIES LINKER_LANGUAGE CXX)