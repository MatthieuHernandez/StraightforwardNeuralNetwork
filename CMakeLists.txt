cmake_minimum_required(VERSION 3.10)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("-- Compile in ${CMAKE_BUILD_TYPE} mode")

if(MSVC)
  message("-- CMake run for msvc")
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
	add_compile_options(/DNDEBUG /O2 /Ot /Ob3 /Oi /Oy /GR /GT /GL /GS- /MT /fp:fast)
  else()
	add_compile_options(/Od /sdl /GR /GS /Zi /MTd)
  endif()
  add_link_options(/LTCG)

  add_link_options(/NODEFAULTLIB:libboost_serialization-vc143-mt-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-s-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-gd-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-sgd-x64-1_80.lib)
endif()

add_subdirectory(src)
add_subdirectory(tests)

if(MSVC)
  set_target_properties(DatasetTests PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)../../../Build")
  target_compile_options(StraightforwardNeuralNetwork PUBLIC /openmp:experimental /W4 /WX)
  target_link_options(StraightforwardNeuralNetwork PUBLIC /LTCG)
  target_compile_options(Boost PRIVATE /W0)
else()
  target_compile_options(StraightforwardNeuralNetwork PUBLIC $<$<CONFIG:Debug>:-g>
                                                             $<$<CONFIG:Release>:-O3>
                                                             $<$<CONFIG:Release>:-ffast-math>
                                                             -fopenmp
                                                             -Wall
                                                             -Wextra
                                                             -pedantic
                                                             -Werror
                                                             -Wno-unused-parameter)
  if(WIN32)
    target_compile_options(StraightforwardNeuralNetwork PUBLIC -DBOOST_ALL_NO_LIB)
  endif()
  target_compile_options(Boost INTERFACE -w)
  target_compile_options(gtest INTERFACE -w)
endif()
