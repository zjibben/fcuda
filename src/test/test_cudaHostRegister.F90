#include "f90_assert.fpp"

program test_cudaHostRegister

  use fcuda
  use,intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: N = 20
  integer, allocatable :: a(:)
  integer(int64) :: nbytes
  integer :: ierr

  allocate(a(N))

  nbytes = int(N*(storage_size(a)/8), int64)
  call fcudaHostRegister(a, nbytes, cudaHostRegisterDefault, ierr); ASSERT(ierr == 0)
  call fcudaHostUnregister(a, ierr); ASSERT(ierr == 0)

end program test_cudaHostRegister
