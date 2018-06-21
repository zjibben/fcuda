include(`replicator.m4')dnl
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
REPLICATE_INTERFACE_TYPE_DIM(`fcudaHostRegister')
  end interface fcudaHostRegister

  interface fcudaHostUnregister
REPLICATE_INTERFACE_TYPE_DIM(`fcudaHostUnregister')
  end interface fcudaHostUnregister

contains

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaHostRegister_$1(array, size, flags, ierr)
    $2, intent(in), contiguous, target :: array$3
    integer(int64), intent(in) :: size
    integer, intent(in) :: flags
    integer, intent(out) :: ierr
    ierr = cudaHostRegister(c_loc(array), size, flags)
  end subroutine fcudaHostRegister_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

define(`ROUTINE_INSTANCE', `dnl
  subroutine fcudaHostUnregister_$1(array, ierr)
    $2, intent(in), contiguous, target :: array$3
    integer, intent(out) :: ierr
    ierr = cudaHostUnregister(c_loc(array))
  end subroutine fcudaHostUnregister_$1')dnl
REPLICATE_ROUTINE_TYPE_DIM()

end module fcudaHostRegister_function
