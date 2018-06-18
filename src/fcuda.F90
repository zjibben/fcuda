#include "f90_assert.fpp"

module fcuda

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
  use,intrinsic :: iso_c_binding, only: c_loc
  use,intrinsic :: iso_c_binding, only: c_ptr
  use,intrinsic :: iso_c_binding, only: fcuda_dev_ptr => c_ptr
  use cuda_c_binding
  implicit none
  private

  public :: fcudaGetDevice
  public :: fcudaGetDeviceCount
  public :: fcudaGetDeviceProperties

  public :: fcudaMalloc
  public :: fcudaFree

  !! memcpy
  public :: fcudaMemcpy
  interface fcudaMemcpy
    module procedure fcudaMemcpy_dev_dev
    module procedure fcudaMemcpy_dev_rank1, fcudaMemcpy_rank1_dev
  end interface fcudaMemcpy

  !! types
  public :: cudaDeviceProp, fcuda_dev_ptr
  public :: & !cudaMemcpyHostToHost, (currently not implemented)
      cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, &
      cudaMemcpyDeviceToDevice, cudaMemcpyDefault

contains

  !! device query
  subroutine fcudaGetDevice(device, ierr)
    integer, intent(out) :: device, ierr
    ierr = cudaGetDevice(device)
  end subroutine fcudaGetDevice

  subroutine fcudaGetDeviceCount(ndevices, ierr)
    integer, intent(out) :: ndevices, ierr
    ierr = cudaGetDeviceCount(ndevices)
  end subroutine fcudaGetDeviceCount

  subroutine fcudaGetDeviceProperties(prop, device, ierr)
    type(cudaDeviceProp), intent(out) :: prop
    integer, intent(in) :: device
    integer, intent(out) :: ierr
    ierr = cudaGetDeviceProperties(prop, device)
  end subroutine fcudaGetDeviceProperties

  !! malloc
  subroutine fcudaMalloc(devPtr, size, ierr)
    type(fcuda_dev_ptr), intent(out) :: devPtr
    integer(int64), intent(in) :: size
    integer, intent(out) :: ierr
    ierr = cudaMalloc(devPtr, size)
  end subroutine fcudaMalloc

  subroutine fcudaFree(devPtr, ierr)
    type(fcuda_dev_ptr), intent(inout) :: devPtr
    integer, intent(out) :: ierr
    ierr = cudaFree(devPtr)
  end subroutine fcudaFree

  !! memcpy
  subroutine fcudaMemcpy_dev_rank1(dst, src, count, kind, ierr)
    type(fcuda_dev_ptr) :: dst
    class(*), intent(in), target :: src(:)
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    select type (src)
    type is (integer)
      ierr = cudaMemcpy(dst, c_loc(src), count, kind)
    type is (integer(int64))
      ierr = cudaMemcpy(dst, c_loc(src), count, kind)
    type is (real)
      ierr = cudaMemcpy(dst, c_loc(src), count, kind)
    type is (real(r8))
      ierr = cudaMemcpy(dst, c_loc(src), count, kind)
    class default
      INSIST(.false.)
    end select
  end subroutine fcudaMemcpy_dev_rank1

  subroutine fcudaMemcpy_rank1_dev(dst, src, count, kind, ierr)
    class(*), intent(out), target :: dst(:)
    type(fcuda_dev_ptr) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    select type (dst)
    type is (integer)
      ierr = cudaMemcpy(c_loc(dst), src, count, kind)
    type is (integer(int64))
      ierr = cudaMemcpy(c_loc(dst), src, count, kind)
    type is (real)
      ierr = cudaMemcpy(c_loc(dst), src, count, kind)
    type is (real(r8))
      ierr = cudaMemcpy(c_loc(dst), src, count, kind)
    class default
      INSIST(.false.)
    end select
  end subroutine fcudaMemcpy_rank1_dev

  subroutine fcudaMemcpy_dev_dev(dst, src, count, kind, ierr)
    type(fcuda_dev_ptr) :: dst
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(dst, src, count, kind)
  end subroutine fcudaMemcpy_dev_dev

end module fcuda
