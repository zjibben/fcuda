#include "f90_assert.fpp"

module compute_data_kernel

  implicit none
  private

  public :: compute_data

  interface
    subroutine compute_data(grid_size, block_size, stream, a, x, N) &
        bind(c, name="ext_compute_data")
      use,intrinsic :: iso_c_binding, only: c_int, c_ptr
      use fcuda, only: fcudaStream
      integer(c_int), value :: grid_size, block_size, x, N
      type(fcudaStream), value :: stream
      type(c_ptr), value :: a
    end subroutine compute_data
  end interface

end module compute_data_kernel

program test_asynchronous_operations

  use,intrinsic :: iso_fortran_env, only: int64
  use fcuda
  use compute_data_kernel
  implicit none

  integer, parameter :: BLOCK_SIZE = 256
  integer, parameter :: NUM_ITERATIONS = 20
  integer, parameter :: N = 2048*2048 ! batch size
  integer, parameter :: GRID_SIZE = (N + BLOCK_SIZE-1)/BLOCK_SIZE
  integer(int64), parameter :: NBYTES = N * (storage_size(N) / 8)
  integer, parameter :: X = 5000
  integer, parameter :: NUM_PARALLEL = 5

  integer :: iter, ierr, i, ip2, im2, ip1, im1
  integer :: dH(N,NUM_PARALLEL), d_ex(N,NUM_PARALLEL)
  type(fcuda_dev_ptr) :: dD(NUM_PARALLEL)
  type(fcudaStream) :: stream(NUM_PARALLEL)

  do i = 1,NUM_PARALLEL
    call fcudaStreamCreate(stream(i), ierr); INSIST(ierr == 0)
    call fcudaMalloc(dD(i), NBYTES, ierr); INSIST(ierr == 0)
    call fcudaHostRegister(dH(:,i), NBYTES, cudaHostRegisterDefault, ierr); INSIST(ierr == 0)
  end do

  do iter = -1, NUM_ITERATIONS+2
    i =   modulo(iter     - 1, NUM_PARALLEL) + 1
    ip2 = modulo(iter + 2 - 1, NUM_PARALLEL) + 1
    im2 = modulo(iter - 2 - 1, NUM_PARALLEL) + 1
    ip1 = modulo(iter + 1 - 1, NUM_PARALLEL) + 1
    im1 = modulo(iter - 1 - 1, NUM_PARALLEL) + 1

    if (iter+1 <= NUM_ITERATIONS .and. iter+1 > 0) then
      call fcudaMemcpyAsync(dD(ip1), dH(:,ip1), NBYTES, cudaMemcpyHostToDevice, stream(ip1), ierr)
      INSIST(ierr == 0)
    end if

    if (iter <= NUM_ITERATIONS .and. iter > 0) &
        call compute_data(GRID_SIZE, BLOCK_SIZE, stream(i), dD(i), X, N)

    if (iter-1 <= NUM_ITERATIONS .and. iter-1 > 0) then
      call fcudaMemcpyAsync(dH(:,im1), dD(im1), NBYTES, cudaMemcpyDeviceToHost, stream(im1), ierr)
      INSIST(ierr == 0)
    end if

    ! cpu stuff
    if (iter-2 <= NUM_ITERATIONS .and. iter-2 > 0) then
      call fcudaStreamSynchronize(stream(im2), ierr); INSIST(ierr == 0)
      call compare_data(N, dH(:,im2), d_ex(:,im2))
    end if

    if (iter+2 <= NUM_ITERATIONS .and. iter+2 > 0) &
        call generate_test_data(N, X, dH(:,ip2), d_ex(:,ip2))
  end do

  do i = 1, NUM_PARALLEL
    call fcudaStreamDestroy(stream(i), ierr); INSIST(ierr == 0)
    call fcudaFree(dD(i), ierr); INSIST(ierr == 0)
    call fcudaHostUnregister(dH(:,i), ierr); INSIST(ierr == 0)
  end do

  call fcudaDeviceReset(ierr)

contains

  subroutine generate_test_data(N, X, input, output)

    integer, intent(in) :: N, X
    integer, intent(out) :: input(:), output(:)

    integer :: i

    do i = 1, N
      input(i) = i
      output(i) = input(i) * X
    end do

  end subroutine generate_test_data

  subroutine compare_data(N, a, b)

    integer, intent(in) :: N, a(:), b(:)

    integer :: i
    logical :: different

    write(*,'(a)',advance='no') "Comparing data...  "
    do i = 1, N
      different = a(i) /= b(i)
      if (different) then
        print *, i, a(i), b(i)
        exit
      end if
    end do

    if (different) then
      print *, "Arrays do not match."
    else
      print *, "Arrays match."
    end if

    INSIST(.not.different)

  end subroutine compare_data

end program test_asynchronous_operations
