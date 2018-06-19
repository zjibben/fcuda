#ifdef _TYPE_
#ifdef _DIM_
#ifdef fcudaHostRegister_type
#ifdef fcudaHostUnregister_type

subroutine fcudaHostRegister_type(array, size, flags, ierr)
  _TYPE_, intent(in), target :: array _DIM_
  integer(int64), intent(in) :: size
  integer, intent(in) :: flags
  integer, intent(out) :: ierr
  ierr = cudaHostRegister(c_loc(array), size, flags)
end subroutine fcudaHostRegister_type

subroutine fcudaHostUnregister_type(array, ierr)
  _TYPE_, intent(in), target :: array _DIM_
  integer, intent(out) :: ierr
  ierr = cudaHostUnregister(c_loc(array))
end subroutine fcudaHostUnregister_type

#endif
#endif
#endif
#endif

#undef _TYPE_
#undef _DIM_
#undef fcudaHostRegister_type
#undef fcudaHostUnregister_type
