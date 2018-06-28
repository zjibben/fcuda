include(`replicator.m4')dnl
module fcudaMemcpy_function

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
  use,intrinsic :: iso_c_binding, only: c_loc
  use,intrinsic :: iso_c_binding, only: fcuda_dev_ptr => c_ptr
  use cuda_c_binding
  implicit none
  private

  public :: fcudaMemcpy

  interface fcudaMemcpy
    module procedure fcudaMemcpy_dev_dev

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpy_HtD')

REPLICATE_INTERFACE_DIM(`fcudaMemcpy_HtD', `ptr')

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpy_DtH')

REPLICATE_INTERFACE_DIM(`fcudaMemcpy_DtH', `ptr')

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpy_HtH')

REPLICATE_INTERFACE_DIM(`fcudaMemcpy_HtH', `ptr')
  end interface fcudaMemcpy

contains

  subroutine fcudaMemcpy_dev_dev(dst, src, count, kind, ierr)
    type(fcuda_dev_ptr) :: dst
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(dst, src, count, kind)
  end subroutine fcudaMemcpy_dev_dev

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpy_DtH_$1(dst, src, count, kind, ierr)
    $2, intent(out), contiguous, target :: dst$3
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(c_loc(dst), src, count, kind)
  end subroutine fcudaMemcpy_DtH_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

REPLICATE_ROUTINE_DIM(`ptr', `type(fcuda_dev_ptr)')

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpy_HtD_$1(dst, src, count, kind, ierr)
    type(fcuda_dev_ptr) :: dst
    $2, intent(in), contiguous, target :: src$3
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(dst, c_loc(src), count, kind)
  end subroutine fcudaMemcpy_HtD_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

REPLICATE_ROUTINE_DIM(`ptr', `type(fcuda_dev_ptr)')

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpy_HtH_$1(dst, src, count, kind, ierr)
    $2, intent(out), contiguous, target :: dst$3
    $2, intent(in), contiguous, target :: src$3
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(c_loc(dst), c_loc(src), count, kind)
  end subroutine fcudaMemcpy_HtH_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

REPLICATE_ROUTINE_DIM(`ptr', `type(fcuda_dev_ptr)')

end module fcudaMemcpy_function
