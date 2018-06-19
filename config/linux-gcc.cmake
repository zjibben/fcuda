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
