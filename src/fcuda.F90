#include "f90_assert.fpp"

module fcuda

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_c_binding, only: c_ptr
  use,intrinsic :: iso_c_binding, only: fcuda_dev_ptr => c_ptr
  use,intrinsic :: iso_c_binding, only: fcudaStream => c_ptr
  use cuda_c_binding
  use fcudaMemcpy_function
  use fcudaMemcpyAsync_function
  use fcudaHostRegister_function
  implicit none
  private

  public :: fcudaGetDevice
  public :: fcudaGetDeviceCount
  public :: fcudaGetDeviceProperties
  public :: fcudaDeviceReset
  public :: fcudaDeviceSynchronize
  public :: fcudaGetLastError
  public :: fcudaMemGetInfo
  public :: fcudaDeviceGetLimit
  public :: fcudaDeviceSetLimit
  public :: fcudaDeviceSetCacheConfig

  public :: fcudaMalloc
  public :: fcudaFree
  public :: fcudaMemcpy
  public :: fcudaMemcpyAsync
  public :: fcudaHostRegister
  public :: fcudaHostUnregister

  public :: fcudaStreamCreate
  public :: fcudaStreamDestroy
  public :: fcudaStreamSynchronize

  !! types
  public :: cudaDeviceProp
  public :: fcuda_dev_ptr
  public :: fcudaStream
  public :: cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault, &
      cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
  public :: cudaHostRegisterDefault, cudaHostRegisterPortable, &
      cudaHostRegisterMapped, cudaHostRegisterIoMemory
  public :: cudaLimitStackSize, cudaLimitPrintFifoSize, cudaLimitMallocHeapSize, &
      cudaLimitDevRuntimeSyncDepth, cudaLimitDevRuntimePendingLaunchCount

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

  subroutine fcudaDeviceSynchronize(ierr)
    integer, intent(out) :: ierr
    ierr = cudaDeviceSynchronize()
  end subroutine fcudaDeviceSynchronize

  subroutine fcudaGetLastError(ierr)
    integer, intent(out) :: ierr
    ierr = cudaGetLastError()
  end subroutine fcudaGetLastError

  subroutine fcudaMemGetInfo(free, total, ierr)
    integer(int64), intent(out) :: free, total
    integer, intent(out) :: ierr
    ierr = cudaMemGetInfo(free, total)
  end subroutine fcudaMemGetInfo

  subroutine fcudaDeviceGetLimit(pval, limit, ierr)
    integer(int64), intent(out) :: pval
    integer, intent(in) :: limit
    integer, intent(out) :: ierr
    ierr = cudaDeviceGetLimit(pval, limit)
  end subroutine fcudaDeviceGetLimit

  subroutine fcudaDeviceSetLimit(limit, pval, ierr)
    integer, intent(in) :: limit
    integer(int64), intent(in) :: pval
    integer, intent(out) :: ierr
    ierr = cudaDeviceSetLimit(limit, pval)
  end subroutine fcudaDeviceSetLimit
  
  subroutine fcudaDeviceSetCacheConfig(cache_config, ierr)
    integer, intent(in) :: cache_config
    integer, intent(out) :: ierr
    ierr = cudaDeviceSetCacheConfig(cache_config)
  end subroutine fcudaDeviceSetCacheConfig

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

  !! streams
  subroutine fcudaStreamCreate(pstream, ierr)
    type(fcudaStream), intent(out) :: pstream
    integer, intent(out) :: ierr
    ierr = cudaStreamCreate(pstream)
  end subroutine fcudaStreamCreate

  subroutine fcudaStreamDestroy(pstream, ierr)
    type(fcudaStream), intent(inout) :: pstream
    integer, intent(out) :: ierr
    ierr = cudaStreamDestroy(pstream)
  end subroutine fcudaStreamDestroy

  subroutine fcudaStreamSynchronize(pstream, ierr)
    type(fcudaStream), intent(in) :: pstream
    integer, intent(out) :: ierr
    ierr = cudaStreamSynchronize(pstream)
  end subroutine fcudaStreamSynchronize

end module fcuda
