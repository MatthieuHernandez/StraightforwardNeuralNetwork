cmake_minimum_required(VERSION 3.10)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("-- Compile in ${CMAKE_BUILD_TYPE} mode")

if(MSVC)
  message("-- CMake run for msvc")
  add_compile_options($<$<CONFIG:Debug>:/Od>
                      $<$<CONFIG:Debug>:/sdl>
                      $<$<CONFIG:Debug>:/GR>
                      $<$<CONFIG:Debug>:/GS>
                      $<$<CONFIG:Debug>:/Zi>
                      $<$<CONFIG:Debug>:/MTd>
                      $<$<CONFIG:Release>:/DNDEBUG>
                      $<$<CONFIG:Release>:/O2>
                      $<$<CONFIG:Release>:/Ot>
                      $<$<CONFIG:Release>:/Ob3>
                      $<$<CONFIG:Release>:/Oi>
                      $<$<CONFIG:Release>:/Oy>
                      $<$<CONFIG:Release>:/GR>
                      $<$<CONFIG:Release>:/GT>
                      $<$<CONFIG:Release>:/GL>
                      $<$<CONFIG:Release>:/GS->
                      $<$<CONFIG:Release>:/MT>
                      $<$<CONFIG:Release>:/fp:fast>)
  add_link_options(/LTCG)

  add_link_options(/NODEFAULTLIB:libboost_serialization-vc143-mt-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-s-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-gd-x64-1_80.lib
                   /NODEFAULTLIB:libboost_serialization-vc143-mt-sgd-x64-1_80.lib)
endif()

add_subdirectory(src)
add_subdirectory(tests)

if(MSVC)
  target_compile_options(StraightforwardNeuralNetwork PUBLIC /openmp:experimental /W4 /WX)
  target_compile_options(Boost PRIVATE /W0)
else()
  target_compile_options(StraightforwardNeuralNetwork PUBLIC $<$<CONFIG:Debug>:-g>
                                                             $<$<CONFIG:Release>:-O3>
                                                             -fopenmp
                                                             -Wall
                                                             -Wextra
                                                             -pedantic
                                                             -Werror
                                                             -Wno-unused-parameter)
  if(WIN32)
    target_compile_options(StraightforwardNeuralNetwork PUBLIC -DBOOST_ALL_NO_LIB)
  endif()
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    target_compile_options(Boost INTERFACE -w)
    target_compile_options(gtest INTERFACE -w)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(Boost INTERFACE -Wno-everything)
    target_compile_options(gtest INTERFACE -Wno-everything)
  endif()

endif()
