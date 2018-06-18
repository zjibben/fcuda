cmake_minimum_required(VERSION 3.5)

project(fcuda Fortran C CXX CUDA)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(ENABLE_TESTS "Build test programs" ON)
if(ENABLE_TESTS)
  enable_testing()
endif()

find_package(CUDA)

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
  add_definitions(-DNO_2008_LHS_POLY_REALLOC)
  if(CMAKE_Fortran_COMPILER_VERSION MATCHES "17.*")
    add_definitions(-DINTEL_BUG20180222)
  endif()
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES NAG)
  # currently nothing
endif()

add_subdirectory(src)
