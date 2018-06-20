define(`MOD_PROCEDURE', `dnl
    module procedure $1_$2x$3')dnl
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
  subroutine fcudaHostRegister_$3(array, size, flags, ierr)
    $1, intent(in), contiguous, target :: array$2
    integer(int64), intent(in) :: size
    integer, intent(in) :: flags
    integer, intent(out) :: ierr
    ierr = cudaHostRegister(c_loc(array), size, flags)
  end subroutine fcudaHostRegister_$3

  subroutine fcudaHostUnregister_$3(array, ierr)
    $1, intent(in), contiguous, target :: array$2
    integer, intent(out) :: ierr
    ierr = cudaHostUnregister(c_loc(array))
  end subroutine fcudaHostUnregister_$3')dnl
dnl
define(`BASE_FUNCS_ARR', `dnl
BASE_FUNCS(`$1', `(:)', `$2x1')

BASE_FUNCS(`$1', `(:,:)', `$2x2')

BASE_FUNCS(`$1', `(:,:,:)', `$2x3')

BASE_FUNCS(`$1', `(:,:,:,:)', `$2x4')')dnl
dnl
module fcudaHostRegister_function

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
  use,intrinsic :: iso_c_binding, only: c_loc
  use cuda_c_binding
  implicit none
  private

  public :: fcudaHostRegister
  public :: fcudaHostUnregister

  interface fcudaHostRegister
MOD_PROCEDURE_ARR_TYPES(`fcudaHostRegister')
  end interface fcudaHostRegister

  interface fcudaHostUnregister
MOD_PROCEDURE_ARR_TYPES(`fcudaHostUnregister')
  end interface fcudaHostUnregister

contains

BASE_FUNCS_ARR(`integer', `i4')

BASE_FUNCS_ARR(`real', `r4')

BASE_FUNCS_ARR(`real(r8)', `r8')

end module fcudaHostRegister_function
