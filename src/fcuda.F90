#include "f90_assert.fpp"

module fcuda

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_c_binding, only: c_ptr
  use,intrinsic :: iso_c_binding, only: fcuda_dev_ptr => c_ptr
  use cuda_c_binding
  use fcudaMemcpy_function
  use fcuda_register_functions
  implicit none
  private

  public :: fcudaGetDevice
  public :: fcudaGetDeviceCount
  public :: fcudaGetDeviceProperties
  public :: fcudaDeviceReset

  public :: fcudaMalloc
  public :: fcudaFree
  public :: fcudaMemcpy
  public :: fcudaHostRegister
  public :: fcudaHostUnregister

  !! types
  public :: cudaDeviceProp, fcuda_dev_ptr
  public :: cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault, &
      cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
  public :: cudaHostRegisterDefault, cudaHostRegisterPortable, &
      cudaHostRegisterMapped, cudaHostRegisterIoMemory

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

  subroutine fcudaDeviceReset(ierr)
    integer, intent(out) :: ierr
    ierr = cudaDeviceReset()
  end subroutine fcudaDeviceReset

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

end module fcuda
