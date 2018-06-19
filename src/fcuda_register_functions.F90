module fcuda_register_functions

  use,intrinsic :: iso_fortran_env, only: int64
  use,intrinsic :: iso_fortran_env, only: r8 => real64
  use,intrinsic :: iso_c_binding, only: c_loc
  use cuda_c_binding
  implicit none
  private

  public :: fcudaHostRegister
  public :: fcudaHostUnregister

  interface fcudaHostRegister
    module procedure fcudaHostRegister_i4x1, fcudaHostRegister_i4x2, fcudaHostRegister_i4x3
    module procedure fcudaHostRegister_r4x1, fcudaHostRegister_r4x2, fcudaHostRegister_r4x3
    module procedure fcudaHostRegister_r8x1, fcudaHostRegister_r8x2, fcudaHostRegister_r8x3
  end interface fcudaHostRegister

  interface fcudaHostUnregister
    module procedure fcudaHostUnregister_i4x1, fcudaHostUnregister_i4x2, fcudaHostUnregister_i4x3
    module procedure fcudaHostUnregister_r4x1, fcudaHostUnregister_r4x2, fcudaHostUnregister_r4x3
    module procedure fcudaHostUnregister_r8x1, fcudaHostUnregister_r8x2, fcudaHostUnregister_r8x3
  end interface fcudaHostUnregister

contains

#define _TYPE_ integer
#define _DIM_ (:)
#define fcudaHostRegister_type fcudaHostRegister_i4x1
#define fcudaHostUnregister_type fcudaHostUnregister_i4x1
#include "fcudaHostRegister.fpp"

#define _TYPE_ integer
#define _DIM_ (:,:)
#define fcudaHostRegister_type fcudaHostRegister_i4x2
#define fcudaHostUnregister_type fcudaHostUnregister_i4x2
#include "fcudaHostRegister.fpp"

#define _TYPE_ integer
#define _DIM_ (:,:,:)
#define fcudaHostRegister_type fcudaHostRegister_i4x3
#define fcudaHostUnregister_type fcudaHostUnregister_i4x3
#include "fcudaHostRegister.fpp"

#define _TYPE_ real
#define _DIM_ (:)
#define fcudaHostRegister_type fcudaHostRegister_r4x1
#define fcudaHostUnregister_type fcudaHostUnregister_r4x1
#include "fcudaHostRegister.fpp"

#define _TYPE_ real
#define _DIM_ (:,:)
#define fcudaHostRegister_type fcudaHostRegister_r4x2
#define fcudaHostUnregister_type fcudaHostUnregister_r4x2
#include "fcudaHostRegister.fpp"

#define _TYPE_ real
#define _DIM_ (:,:,:)
#define fcudaHostRegister_type fcudaHostRegister_r4x3
#define fcudaHostUnregister_type fcudaHostUnregister_r4x3
#include "fcudaHostRegister.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:)
#define fcudaHostRegister_type fcudaHostRegister_r8x1
#define fcudaHostUnregister_type fcudaHostUnregister_r8x1
#include "fcudaHostRegister.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:,:)
#define fcudaHostRegister_type fcudaHostRegister_r8x2
#define fcudaHostUnregister_type fcudaHostUnregister_r8x2
#include "fcudaHostRegister.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:,:,:)
#define fcudaHostRegister_type fcudaHostRegister_r8x3
#define fcudaHostUnregister_type fcudaHostUnregister_r8x3
#include "fcudaHostRegister.fpp"

end module fcuda_register_functions
