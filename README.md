# FCUDA

The goal of FCUDA is to provide Fortran bindings to CUDA C. PGI provides CUDA
Fortran, but its use requires your entire project be compiled with a
PGI-compatible compiler. Sometimes this is not possible, or otherwise an
undesired limitation. The alternative is to use CUDA C with NVIDIA's `nvcc`
compiler, and to access it through Fortran's ISO C bindings.

## Dependencies

- CMake 3.5 or newer
- CUDA 9 or newer

The following Fortran compilers have been tested:

- ifort 17.0.x and 18.0.x
- gfortran 8.x and 9.x
- NAG 6.2

## Building

FCUDA uses an out-of-source build process. From the root of this repository:

```
$ mkdir build
$ cd build
$ cmake -C ../config/linux-<YOUR-COMPILER>.cmake ..
$ make
```

## Testing

From the build directory:

```
$ ctest
```

## Usage

Example usage can be found in the `src/test` directory. More detailed
instructions forthcoming.
