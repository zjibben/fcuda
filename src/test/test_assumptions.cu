#undef NDEBUG
#include <assert.h>

int main()
{
  assert(sizeof(cudaError_t) == sizeof(int));
  assert(sizeof(cudaStream_t) == sizeof(void*));
  assert(sizeof(long) == sizeof(size_t));
  return 0;
}
