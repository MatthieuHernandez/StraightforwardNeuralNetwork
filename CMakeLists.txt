# execute this from build directory : cmake -G"Unix Makefiles" ./.. && make && ./tests/unit_tests/UnitTests
cmake_minimum_required(VERSION 3.0)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 20)

# Applies CMAKE_CXX_FLAGS to all targets in the current CMake directory.
# Pre: apply_global_cxx_flags_to_all_targets() must be invoked.
macro(apply_global_cxx_flags_to_all_targets)
    separate_arguments(_global_cxx_flags_list UNIX_COMMAND ${CMAKE_CXX_FLAGS})
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target ${_targets})
        target_compile_options(${_target} PUBLIC ${_global_cxx_flags_list})
    endforeach()
    unset(CMAKE_CXX_FLAGS)
    set(_flag_sync_required TRUE)
endmacro()

# Removes the specified compile flag from the specified target.
macro(remove_target_compile_options _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("-- Compile in ${CMAKE_BUILD_TYPE} mode")

if(MSVC)
  message("-- CMake run for msvc")
  string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REPLACE "/MD" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D \"NDEBUG\" /O2 /Ot /Ob3 /Oi /Oy /GR /GT /GL /GS- /MT /fp:fast")
  string(REPLACE "/MDd" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /sdl /GR /GS /Zi /MTd")
  set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc143-mt-x64-1_80.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc143-mt-s-x64-1_80.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc143-mt-gd-x64-1_80.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc143-mt-sgd-x64-1_80.lib")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif()

add_subdirectory(src)
add_subdirectory(tests)

apply_global_cxx_flags_to_all_targets()

if(MSVC)
  set_target_properties(DatasetTests PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)../../../Build")
  target_compile_options(StraightforwardNeuralNetwork PUBLIC /openmp:experimental /W4 /WX)
  target_compile_options(Boost PRIVATE /W0)
  remove_target_compile_options(Boost /W3)
  remove_target_compile_options(StraightforwardNeuralNetwork /W3)
  remove_target_compile_options(UnitTests /W3)
  remove_target_compile_options(DatasetTests /W3)
else()
  target_compile_options(StraightforwardNeuralNetwork PUBLIC -fopenmp -Wall -Wextra -pedantic -Werror -Wno-unused-parameter)
  target_compile_options(Boost PRIVATE -w)
endif()