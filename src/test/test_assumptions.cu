#undef NDEBUG
#include <assert.h>

int main()
{
  assert(sizeof(cudaError_t) == sizeof(int));
  assert(sizeof(long) == sizeof(size_t));
  assert(sizeof(void*) == sizeof(cudaStream_t));

  return 0;
}
