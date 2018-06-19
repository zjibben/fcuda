#undef NDEBUG
#include <assert.h>

int main()
{
  assert(sizeof(long) == sizeof(size_t));
  assert(sizeof(void*) == sizeof(cudaStream_t));

  // Also assuming cuda runtime functions are linked in
  // as long as any file is compiled with nvcc.

  return 0;
}
