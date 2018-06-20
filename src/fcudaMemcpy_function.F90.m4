define(`MOD_PROCEDURE', `dnl
    module procedure $1_dev_$2x$3, $1_$2x$3_dev, $1_$2x$3_$2x$3')dnl
define(`MOD_PROCEDURE_ARR', `dnl
MOD_PROCEDURE(`$1', `$2', `1')
MOD_PROCEDURE(`$1', `$2', `2')
MOD_PROCEDURE(`$1', `$2', `3')
MOD_PROCEDURE(`$1', `$2', `4')')dnl
define(`MOD_PROCEDURE_ARR_TYPES', `dnl
MOD_PROCEDURE_ARR(`$1', `i4')

MOD_PROCEDURE_ARR(`$1', `r4')

MOD_PROCEDURE_ARR(`$1', `r8')')dnl
dnl
dnl
define(`BASE_FUNCS', `dnl
  subroutine fcudaMemcpy_$3_dev(dst, src, count, kind, ierr)
    $1, intent(out), contiguous, target :: dst$2
    type(fcuda_dev_ptr), intent(in) :: src
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(c_loc(dst), src, count, kind)
  end subroutine fcudaMemcpy_$3_dev

  subroutine fcudaMemcpy_dev_$3(dst, src, count, kind, ierr)
    type(fcuda_dev_ptr) :: dst
    $1, intent(in), contiguous, target :: src$2
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(dst, c_loc(src), count, kind)
  end subroutine fcudaMemcpy_dev_$3

  subroutine fcudaMemcpy_$3_$3(dst, src, count, kind, ierr)
    $1, intent(out), contiguous, target :: dst$2
    $1, intent(in), contiguous, target :: src$2
    integer(int64), intent(in) :: count
    integer, intent(in) :: kind
    integer, intent(out) :: ierr
    ierr = cudaMemcpy(c_loc(dst), c_loc(src), count, kind)
  end subroutine fcudaMemcpy_$3_$3')dnl
dnl
define(`BASE_FUNCS_ARR', `dnl
BASE_FUNCS(`$1', `(:)', `$2x1')

BASE_FUNCS(`$1', `(:,:)', `$2x2')

BASE_FUNCS(`$1', `(:,:,:)', `$2x3')

BASE_FUNCS(`$1', `(:,:,:,:)', `$2x4')')dnl
dnl
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

MOD_PROCEDURE_ARR_TYPES(`fcudaMemcpy')
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

BASE_FUNCS_ARR(`integer', `i4')

BASE_FUNCS_ARR(`real', `r4')

BASE_FUNCS_ARR(`real(r8)', `r8')

end module fcudaMemcpy_function
