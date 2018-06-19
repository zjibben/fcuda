#include "f90_assert.fpp"

program test_cudaGetDevice

  use fcuda
  implicit none

  integer :: i, ndevices, ierr
  type(cudaDeviceProp) :: prop

  call fcudaGetDeviceCount(ndevices, ierr)

  ASSERT(ierr==0)
  print *, "Number of devices: ", ndevices

  do i = 0, ndevices-1
    call fcudaGetDeviceProperties(prop, i, ierr)
    ASSERT(ierr==0)
    print *, "Device number: ", i
    print *, "  Device name: ", prop%name(:16)
    print *, "  Memory Clock Rate (Khz): ", prop%memoryClockRate
    print *, "  Memory Bus Width (bits): ", prop%memoryBusWidth
    print *, "  Peak Memory Bandwidth (GB/s): ", 2*prop%memoryClockRate*(prop%memoryBusWidth/8)/1.0e6
  end do

  if (ierr /= 0) stop 1

end program test_cudaGetDevice
