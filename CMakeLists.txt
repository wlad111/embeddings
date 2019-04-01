cmake_minimum_required(VERSION 3.13)
project(embeddings)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(extern/ml_lib)

add_library(embeddings SHARED library.cpp library.h)