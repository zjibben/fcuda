set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_Fortran_COMPILER nagfor CACHE STRING "Fortran compiler")
set(CMAKE_Fortran_FLAGS_DEBUG "-u -g -gline -C -nan" CACHE
    STRING "Flags used by the compiler during debug builds")
set(CMAKE_Fortran_FLAGS_RELEASE "-u -O3 -DNDEBUG" CACHE
    STRING "Flags used by the compiler during release builds")
