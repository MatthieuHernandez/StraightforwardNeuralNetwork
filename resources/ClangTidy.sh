#!/bin/bash

find ../src ../tests ../include \
  -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.tpp" \) \
  ! -path "../src/external_library/*" \
  -exec clang-tidy -p ../build_clang/ -header-filter=.* \
  {} +
