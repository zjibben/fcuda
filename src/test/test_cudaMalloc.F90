#include "f90_assert.fpp"

program test_cudaMalloc

  use fcuda
  use,intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: N = 10
  integer :: ierr, aH(N)
  integer(int64) :: nbytes
  type(fcuda_dev_ptr) :: aD

  nbytes = int(N*(storage_size(aH)/8), int64)

  call fcudaMalloc(aD, nbytes, ierr); INSIST(ierr==0)

  aH = 1
  call fcudaMemcpy(aD, aH, nbytes, cudaMemcpyHostToDevice, ierr); INSIST(ierr==0)

  aH = 0

  call fcudaMemcpy(aH, aD, nbytes, cudaMemcpyDeviceToHost, ierr); INSIST(ierr==0)

  call fcudaFree(aD, ierr); INSIST(ierr==0)
  call fcudaDeviceReset(ierr); INSIST(ierr==0)

  ! check what came back
  INSIST(.not.any(aH /= 1))

end program test_cudaMalloc
