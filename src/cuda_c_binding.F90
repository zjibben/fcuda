module cuda_c_binding

  use,intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int, c_size_t
  implicit none
  private

  !! CUDA Device Queries
  type, bind(c), public :: cudaDeviceProp
    character(kind=c_char) :: name(256)
    integer(c_size_t) :: totalGlobalMem
    integer(c_size_t) :: sharedMemPerBlock
    integer(c_int) :: regsPerBlock
    integer(c_int) :: warpSize
    integer(c_size_t) :: memPitch
    integer(c_int) :: maxThreadsPerBlock
    integer(c_int) :: maxThreadsDim(3)
    integer(c_int) :: maxGridSize(3)
    integer(c_int) :: clockRate
    integer(c_size_t) :: totalConstMem
    integer(c_int) :: major
    integer(c_int) :: minor
    integer(c_size_t) :: textureAlignment
    integer(c_size_t) :: texturePitchAlignment
    integer(c_int) :: deviceOverlap
    integer(c_int) :: multiProcessorCount
    integer(c_int) :: kernelExecTimeoutEnabled
    integer(c_int) :: integrated
    integer(c_int) :: canMapHostMemory
    integer(c_int) :: computeMode
    integer(c_int) :: maxTexture1D
    integer(c_int) :: maxTexture1DMipmap
    integer(c_int) :: maxTexture1DLinear
    integer(c_int) :: maxTexture2D(2)
    integer(c_int) :: maxTexture2DMipmap(2)
    integer(c_int) :: maxTexture2DLinear(3)
    integer(c_int) :: maxTexture2DGather(2)
    integer(c_int) :: maxTexture3D(3)
    integer(c_int) :: maxTexture3DAlt(3)
    integer(c_int) :: maxTextureCubemap
    integer(c_int) :: maxTexture1DLayered(2)
    integer(c_int) :: maxTexture2DLayered(3)
    integer(c_int) :: maxTextureCubemapLayered(2)
    integer(c_int) :: maxSurface1D
    integer(c_int) :: maxSurface2D(2)
    integer(c_int) :: maxSurface3D(3)
    integer(c_int) :: maxSurface1DLayered(2)
    integer(c_int) :: maxSurface2DLayered(3)
    integer(c_int) :: maxSurfaceCubemap
    integer(c_int) :: maxSurfaceCubemapLayered(2)
    integer(c_size_t) :: surfaceAlignment
    integer(c_int) :: concurrentKernels
    integer(c_int) :: ECCEnabled
    integer(c_int) :: pciBusID
    integer(c_int) :: pciDeviceID
    integer(c_int) :: pciDomainID
    integer(c_int) :: tccDriver
    integer(c_int) :: asyncEngineCount
    integer(c_int) :: unifiedAddressing
    integer(c_int) :: memoryClockRate
    integer(c_int) :: memoryBusWidth
    integer(c_int) :: l2CacheSize
    integer(c_int) :: maxThreadsPerMultiProcessor
    integer(c_int) :: streamPrioritiesSupported
    integer(c_int) :: globalL1CacheSupported
    integer(c_int) :: localL1CacheSupported
    integer(c_size_t) :: sharedMemPerMultiprocessor
    integer(c_int) :: regsPerMultiprocessor
    integer(c_int) :: managedMemory
    integer(c_int) :: isMultiGpuBoard
    integer(c_int) :: multiGpuBoardGroupID
    integer(c_int) :: hostNativeAtomicSupported
    integer(c_int) :: singleToDoublePrecisionPerfRatio
    integer(c_int) :: pageableMemoryAccess
    integer(c_int) :: concurrentManagedAccess
    integer(c_int) :: computePreemptionSupported
    integer(c_int) :: canUseHostPointerForRegisteredMem
    integer(c_int) :: cooperativeLaunch
    integer(c_int) :: cooperativeMultiDeviceLaunch
    integer(c_size_t) :: sharedMemPerBlockOptin
    integer(c_int) :: pageableMemoryAccessUsesHostPageTables
    integer(c_int) :: directManagedMemAccessFromHost
  end type cudaDeviceProp

  public :: cudaGetDevice
  public :: cudaGetDeviceCount
  public :: cudaGetDeviceProperties

  interface
    function cudaGetDevice(device) &
        result(ierr) bind(c, name="ext_cudaGetDevice")
      import c_int
      integer(c_int), intent(out) :: device
      integer(c_int) :: ierr
    end function cudaGetDevice
    function cudaGetDeviceCount(ndevices) &
        result(ierr) bind(c, name="ext_cudaGetDeviceCount")
      import c_int
      integer(c_int), intent(out) :: ndevices
      integer(c_int) :: ierr
    end function cudaGetDeviceCount
    function cudaGetDeviceProperties(prop, device) &
        result(ierr) bind(c, name="ext_cudaGetDeviceProperties")
      import c_int, cudaDeviceProp
      type(cudaDeviceProp), intent(out) :: prop
      integer(c_int), intent(in), value :: device
      integer(c_int) :: ierr
    end function cudaGetDeviceProperties
  end interface

  !! CUDA Malloc
  public :: cudaMalloc
  public :: cudaFree

  interface
    function cudaMalloc(devPtr, size) &
        result(ierr) bind(c, name="ext_cudaMalloc")
      import c_ptr, c_size_t, c_int
      type(c_ptr) :: devPtr
      integer(c_size_t), value :: size
      integer(c_int) :: ierr
    end function cudaMalloc
    function cudaFree(devPtr) &
        result(ierr) bind(c, name="ext_cudaFree")
      import c_ptr, c_size_t, c_int
      type(c_ptr), value :: devPtr
      integer(c_int) :: ierr
    end function cudaFree
  end interface

  !! memcpy
  enum, bind(c)
    enumerator :: &
        cudaMemcpyHostToHost     = 0, &
        cudaMemcpyHostToDevice   = 1, &
        cudaMemcpyDeviceToHost   = 2, &
        cudaMemcpyDeviceToDevice = 3, &
        cudaMemcpyDefault        = 4
  end enum
  public :: cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, &
      cudaMemcpyDeviceToDevice, cudaMemcpyDefault

  public :: cudaMemcpy

  interface
    function cudaMemcpy(dst, src, count, kind) &
        result(ierr) bind(c, name="ext_cudaMemcpy")
      import c_ptr, c_size_t, c_int
      type(c_ptr), value :: dst
      type(c_ptr), intent(in), value :: src
      integer(c_size_t), value :: count
      integer(c_int), value :: kind
    end function cudaMemcpy
  end interface

end module cuda_c_binding
