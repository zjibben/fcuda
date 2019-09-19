set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_Fortran_COMPILER gfortran CACHE STRING "Fortran compiler")
set(CMAKE_Fortran_FLAGS "-std=f2008ts -pedantic -Wall -Wextra -ffree-line-length-none -frealloc-lhs \
-Wno-compare-reals -Wno-integer-division"
    CACHE STRING "Fortran compile flags")
set(CMAKE_Fortran_FLAGS_DEBUG "-g ${CMAKE_Fortran_FLAGS}"
  CACHE STRING "Flags used by the compiler during debug builds")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG ${CMAKE_Fortran_FLAGS}"
  CACHE STRING "Fortran compile flags")

# This line is needed to make nvcc compile with GCC 9.x.
# Can't only apply this line if GCC 9.x is detected, because
# this has to be set before CMake identifies compilers.
set(CMAKE_CUDA_FLAGS "-U__GNUC__ -D__GNUC__=8 ${CMAKE_CUDA_FLAGS}" CACHE
  STRING "CUDA compile flags")
