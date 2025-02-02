#!/bin/bash
set -e
# ../src ../tests ../include 
find ../examples \
  -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.tpp" \) \
  ! -path "../src/external_library/*" \
  ! -path "../tests/audiofile/*" \
  -exec clang-tidy -p ../build_clang/ \
  {} +