set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER icc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER icpc CACHE STRING "C++ compiler")
set(CMAKE_Fortran_COMPILER ifort CACHE STRING "Fortran compiler")
set(CMAKE_Fortran_FLAGS_DEBUG "-u -g -check all,noarg_temp_created -init=snan -traceback -fpe0"
    CACHE STRING "Flags used by the compiler during debug builds")
set(CMAKE_Fortran_FLAGS_RELEASE "-g -u -O3 -fpe0 -DNDEBUG" CACHE
    STRING "Flags used by the compiler during release builds")
