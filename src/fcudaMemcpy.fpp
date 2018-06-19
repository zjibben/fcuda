#ifdef _TYPE_
#ifdef _DIM_
#ifdef fcudaMemcpy_dev_host
#ifdef fcudaMemcpy_host_dev
#ifdef fcudaMemcpy_host_host

subroutine fcudaMemcpy_host_dev(dst, src, count, kind, ierr)
  _TYPE_, intent(out), target :: dst _DIM_
  type(fcuda_dev_ptr) :: src
  integer(int64), intent(in) :: count
  integer, intent(in) :: kind
  integer, intent(out) :: ierr
  ierr = cudaMemcpy(c_loc(dst), src, count, kind)
end subroutine fcudaMemcpy_host_dev

subroutine fcudaMemcpy_dev_host(dst, src, count, kind, ierr)
  type(fcuda_dev_ptr) :: dst
  _TYPE_, intent(in), target :: src _DIM_
  integer(int64), intent(in) :: count
  integer, intent(in) :: kind
  integer, intent(out) :: ierr
  ierr = cudaMemcpy(dst, c_loc(src), count, kind)
end subroutine fcudaMemcpy_dev_host

subroutine fcudaMemcpy_host_host(dst, src, count, kind, ierr)
  _TYPE_, intent(out), target :: dst _DIM_, src _DIM_
  integer(int64), intent(in) :: count
  integer, intent(in) :: kind
  integer, intent(out) :: ierr
  ierr = cudaMemcpy(c_loc(dst), c_loc(src), count, kind)
end subroutine fcudaMemcpy_host_host

#endif
#endif
#endif
#endif
#endif

#undef _TYPE_
#undef _DIM_
#undef fcudaMemcpy_dev_host
#undef fcudaMemcpy_host_dev
#undef fcudaMemcpy_host_host
