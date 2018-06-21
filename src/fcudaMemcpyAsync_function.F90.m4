include(`replicator.m4')dnl
module fcudaMemcpyAsync_function

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
  use,intrinsic :: iso_c_binding, only: c_loc
  use,intrinsic :: iso_c_binding, only: fcuda_dev_ptr => c_ptr
  use,intrinsic :: iso_c_binding, only: fcudaStream => c_ptr
  use cuda_c_binding
  implicit none
  private

  ! WARN: Errors are likely to occur if copy-in/copy-out is used
  !       on arguments to fcudaMemcpyAsync (e.g., non-contiguous
  !       arrays). The is_contiguous f08 intrinsic should detect
  !       this, but is not yet supported by gfortran.

  public :: fcudaMemcpyAsync

  interface fcudaMemcpyAsync
    module procedure fcudaMemcpyAsync_dev_dev

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpyAsync_HtD')

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpyAsync_DtH')

REPLICATE_INTERFACE_TYPE_DIM(`fcudaMemcpyAsync_HtH')
  end interface fcudaMemcpyAsync

contains

  subroutine fcudaMemcpyAsync_dev_dev(dst, src, count, kind, stream, ierr)
    type(fcuda_dev_ptr) :: dst
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    type(fcudaStream), intent(in) :: stream
    integer, intent(out) :: ierr
    ierr = cudaMemcpyAsync(dst, src, count, kind, stream)
  end subroutine fcudaMemcpyAsync_dev_dev

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpyAsync_DtH_$1(dst, src, count, kind, stream, ierr)
    $2, intent(inout), target :: dst$3
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    type(fcudaStream), intent(in) :: stream
    integer, intent(out) :: ierr
#ifndef __GFORTRAN__
    if (.not.is_contiguous(dst)) then
      ierr = -1
      return
    end if
#endif
    ierr = cudaMemcpyAsync(c_loc(dst), src, count, kind, stream)
  end subroutine fcudaMemcpyAsync_DtH_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpyAsync_HtD_$1(dst, src, count, kind, stream, ierr)
    type(fcuda_dev_ptr) :: dst
    $2, intent(in), target :: src$3
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    type(fcudaStream), intent(in) :: stream
    integer, intent(out) :: ierr
#ifndef __GFORTRAN__
    if (.not.is_contiguous(src)) then
      ierr = -1
      return
    end if
#endif
    ierr = cudaMemcpyAsync(dst, c_loc(src), count, kind, stream)
  end subroutine fcudaMemcpyAsync_HtD_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaMemcpyAsync_HtH_$1(dst, src, count, kind, stream, ierr)
    $2, intent(inout), target :: dst$3
    $2, intent(in), target :: src$3
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    type(fcudaStream), intent(in) :: stream
    integer, intent(out) :: ierr
#ifndef __GFORTRAN__
    if (.not.(is_contiguous(dst) .and. is_contiguous(src))) then
      ierr = -1
      return
    end if
#endif
    ierr = cudaMemcpyAsync(c_loc(dst), c_loc(src), count, kind, stream)
  end subroutine fcudaMemcpyAsync_HtH_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

end module fcudaMemcpyAsync_function
