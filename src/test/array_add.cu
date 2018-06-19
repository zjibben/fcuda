// c = a + b
__global__ void array_add(double* a, double* b, double* c, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N) return;

  c[idx] = a[idx] + b[idx];

  for (int i=idx; i < N; i+=32)
    c[i] = a[i] + b[i];
}

extern "C" void ext_array_add(int grid_size, int block_size,
                              double* a, double* b, double* c, int N)
{
  array_add<<<grid_size,block_size>>>(a, b, c, N);
}
