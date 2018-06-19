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

    module procedure fcudaMemcpy_dev_i4x1, fcudaMemcpy_i4x1_dev, fcudaMemcpy_i4x1_i4x1
    module procedure fcudaMemcpy_dev_i4x2, fcudaMemcpy_i4x2_dev, fcudaMemcpy_i4x2_i4x2
    module procedure fcudaMemcpy_dev_i4x3, fcudaMemcpy_i4x3_dev, fcudaMemcpy_i4x3_i4x3

    module procedure fcudaMemcpy_dev_r4x1, fcudaMemcpy_r4x1_dev, fcudaMemcpy_r4x1_r4x1
    module procedure fcudaMemcpy_dev_r4x2, fcudaMemcpy_r4x2_dev, fcudaMemcpy_r4x2_r4x2
    module procedure fcudaMemcpy_dev_r4x3, fcudaMemcpy_r4x3_dev, fcudaMemcpy_r4x3_r4x3

    module procedure fcudaMemcpy_dev_r8x1, fcudaMemcpy_r8x1_dev, fcudaMemcpy_r8x1_r8x1
    module procedure fcudaMemcpy_dev_r8x2, fcudaMemcpy_r8x2_dev, fcudaMemcpy_r8x2_r8x2
    module procedure fcudaMemcpy_dev_r8x3, fcudaMemcpy_r8x3_dev, fcudaMemcpy_r8x3_r8x3
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

#define _TYPE_ integer
#define _DIM_ (:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_i4x1
#define fcudaMemcpy_host_dev fcudaMemcpy_i4x1_dev
#define fcudaMemcpy_host_host fcudaMemcpy_i4x1_i4x1
#include "fcudaMemcpy.fpp"

#define _TYPE_ integer
#define _DIM_ (:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_i4x2
#define fcudaMemcpy_host_dev fcudaMemcpy_i4x2_dev
#define fcudaMemcpy_host_host fcudaMemcpy_i4x2_i4x2
#include "fcudaMemcpy.fpp"

#define _TYPE_ integer
#define _DIM_ (:,:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_i4x3
#define fcudaMemcpy_host_dev fcudaMemcpy_i4x3_dev
#define fcudaMemcpy_host_host fcudaMemcpy_i4x3_i4x3
#include "fcudaMemcpy.fpp"

#define _TYPE_ real
#define _DIM_ (:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r4x1
#define fcudaMemcpy_host_dev fcudaMemcpy_r4x1_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r4x1_r4x1
#include "fcudaMemcpy.fpp"

#define _TYPE_ real
#define _DIM_ (:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r4x2
#define fcudaMemcpy_host_dev fcudaMemcpy_r4x2_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r4x2_r4x2
#include "fcudaMemcpy.fpp"

#define _TYPE_ real
#define _DIM_ (:,:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r4x3
#define fcudaMemcpy_host_dev fcudaMemcpy_r4x3_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r4x3_r4x3
#include "fcudaMemcpy.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r8x1
#define fcudaMemcpy_host_dev fcudaMemcpy_r8x1_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r8x1_r8x1
#include "fcudaMemcpy.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r8x2
#define fcudaMemcpy_host_dev fcudaMemcpy_r8x2_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r8x2_r8x2
#include "fcudaMemcpy.fpp"

#define _TYPE_ real(r8)
#define _DIM_ (:,:,:)
#define fcudaMemcpy_dev_host fcudaMemcpy_dev_r8x3
#define fcudaMemcpy_host_dev fcudaMemcpy_r8x3_dev
#define fcudaMemcpy_host_host fcudaMemcpy_r8x3_r8x3
#include "fcudaMemcpy.fpp"

end module fcudaMemcpy_function
