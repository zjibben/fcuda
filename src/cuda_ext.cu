//#include <cuda_runtime.h>
//#include <stdio.h>

// device queries
extern "C" int ext_cudaGetDevice(int *device)
{
  return cudaGetDevice(device);
}

extern "C" int ext_cudaGetDeviceCount(int *ndevices)
{
  return cudaGetDeviceCount(ndevices);
}

extern "C" int ext_cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
  return cudaGetDeviceProperties(prop, device);
}

// malloc
extern "C" int ext_cudaMalloc(void** devPtr, size_t size)
{
  return cudaMalloc(devPtr, size);
}

extern "C" int ext_cudaFree(void* devPtr)
{
  return cudaFree(devPtr);
}

// memcpy
extern "C" int ext_cudaMemcpy(void* dst, const void *src, size_t count, cudaMemcpyKind kind)
{
  return cudaMemcpy(dst, src, count, kind);
}
