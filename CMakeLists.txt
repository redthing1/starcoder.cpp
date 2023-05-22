cmake_minimum_required(VERSION 3.12)
PROJECT(starcoder)

# local copy of ggml library
add_library(starcoder_ggml STATIC ggml.c common.cpp)

# starcoder library
add_library(starcoder STATIC starcoder.cpp)
target_link_libraries(starcoder PRIVATE starcoder_ggml)

# starcoder demo main
set(TEST_TARGET starcoder-demo)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE starcoder_ggml starcoder)

#
# starcoder-quantize

set(TEST_TARGET starcoder-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE starcoder_ggml)
