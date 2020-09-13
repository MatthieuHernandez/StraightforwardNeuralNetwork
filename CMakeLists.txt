# execute this from build directory : cmake -G"Unix Makefiles" ./.. && make && ./tests/unit_tests/UnitTests.out
cmake_minimum_required(VERSION 3.0)
project(StraightforwardNeuralNetwork CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  message("-- CMake run for msvc")
  string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D \"NDEBUG\" /O2 /Ot /Ob3 /Oi /Oy /GR /GT /GL /GS- /MT /fp:fast")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /sdl /GR /GS /Zi /MTd")
  set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc142-mt-x64-1_71.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc142-mt-s-x64-1_71.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc142-mt-gd-x64-1_71.lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libboost_serialization-vc142-mt-sgd-x64-1_71.lib")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif()

add_subdirectory(src)
add_subdirectory(tests)

if(MSVC)
  set_target_properties(DatasetTests.out PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)../../../Build")
  target_compile_options(StraightforwardNeuralNetwork BEFORE PUBLIC /W4 /WX)
else()
  target_compile_options(StraightforwardNeuralNetwork BEFORE PUBLIC -Wall -Wextra -pedantic -Werror)
endif()