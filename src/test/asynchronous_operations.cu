#include <stdio.h>

__global__ void compute_data(int *a, int const x, int const n)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx<n) {
    int aa = a[idx];
    int product = 0.0;
    for(int i = 0; i < x; i++) product += aa;
    a[idx] = product;
  }
}

extern "C"
void ext_compute_data(int grid_size, int block_size, cudaStream_t stream,
                      int* a, int const x, int const N)
{
  compute_data<<<grid_size, block_size, 0, stream>>>(a, x, N);
}
