#!/bin/bash
#
clang-tidy -p ./build_clang/ --exclude-header-filter=* --use-color \
  -checks='-*, cppcoreguidelines-owning-memory, readability-avoid-const-params-in-decls' \
  $(find  ./src ./tests ./include ./examples \
  -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.tpp" \) \
  ! -path "./src/external_library/*" \
  ! -path "./tests/audiofile/*")
