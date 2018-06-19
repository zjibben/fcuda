#include "f90_assert.fpp"

module array_add_kernel

  use,intrinsic :: iso_c_binding, only: c_int, c_ptr
  implicit none
  private

  public :: array_add

  interface
    subroutine array_add(grid_size, block_size, a, b, c, N) &
        bind(c, name="ext_array_add")
      import c_ptr, c_int
      integer(c_int), value :: grid_size, block_size, N
      type(c_ptr), value :: a, b, c
    end subroutine array_add
  end interface

end module array_add_kernel

program test_array_add

  use fcuda
  use array_add_kernel
  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
#ifdef NAGFOR
  use,intrinsic :: f90_unix, only: exit
#endif
  implicit none

  integer, parameter :: N = 2053
  integer, parameter :: BLOCK_SIZE = 512
  integer :: ierr, grid_size
  integer(int64) :: nbytes
  real(r8) :: aH(N), bH(N), cH(N)
  type(fcuda_dev_ptr) :: aD, bD, cD

  !! generate test data
  call random_number(aH)
  call random_number(bH)

  !! allocate GPU memory and copy to the GPU
  nbytes = int(N*(storage_size(aH)/8), int64)

  call fcudaMalloc(aD, nbytes, ierr); ASSERT(ierr==0)
  call fcudaMalloc(bD, nbytes, ierr); ASSERT(ierr==0)
  call fcudaMalloc(cD, nbytes, ierr); ASSERT(ierr==0)

  call fcudaMemcpy(aD, aH, nbytes, cudaMemcpyHostToDevice, ierr); ASSERT(ierr==0)
  call fcudaMemcpy(bD, bH, nbytes, cudaMemcpyHostToDevice, ierr); ASSERT(ierr==0)

  !! run the kernel
  grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE
  call array_add(grid_size, BLOCK_SIZE, aD, bD, cD, N)

  !! copy the result out
  call fcudaMemcpy(cH, cD, nbytes, cudaMemcpyDeviceToHost, ierr); ASSERT(ierr==0)

  !! free up
  call fcudaFree(aD, ierr); ASSERT(ierr==0)
  call fcudaFree(bD, ierr); ASSERT(ierr==0)
  call fcudaFree(cD, ierr); ASSERT(ierr==0)

  !! check what came back
  print *, 'max error: ', maxval(abs(cH - (aH + bH)))
  if (any(cH /= aH + bH)) ierr = 1

  call exit(ierr)

end program test_array_add
