#!/bin/bash
#
clang-tidy -p ./build_clang/ --exclude-header-filter=* --use-color --quiet --fix \
  -checks='-*, performance-for-range-copy, google-build-using-namespace' \
  $(find  ./src ./tests ./include ./examples \
  -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.tpp" \) \
  ! -path "./src/external_library/*" \
  ! -path "./tests/audiofile/*")
