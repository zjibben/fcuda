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
  public :: cudaDeviceReset
  public :: cudaDeviceSynchronize
  public :: cudaGetLastError
  public :: cudaMemGetInfo
  public :: cudaDeviceGetLimit
  public :: cudaDeviceSetLimit

  enum, bind(c)
    enumerator :: &
        cudaLimitStackSize = 0, &
        cudaLimitPrintFifoSize = 1, &
        cudaLimitMallocHeapSize = 2, &
        cudaLimitDevRuntimeSyncDepth = 3, &
        cudaLimitDevRuntimePendingLaunchCount = 4
  end enum
  public :: cudaLimitStackSize, cudaLimitPrintFifoSize, cudaLimitMallocHeapSize, &
      cudaLimitDevRuntimeSyncDepth, cudaLimitDevRuntimePendingLaunchCount

  interface
    function cudaGetDevice(device) &
        result(ierr) bind(c, name="cudaGetDevice")
      import c_int
      integer(c_int), intent(out) :: device
      integer(c_int) :: ierr
    end function cudaGetDevice
    function cudaGetDeviceCount(ndevices) &
        result(ierr) bind(c, name="cudaGetDeviceCount")
      import c_int
      integer(c_int), intent(out) :: ndevices
      integer(c_int) :: ierr
    end function cudaGetDeviceCount
    function cudaGetDeviceProperties(prop, device) &
        result(ierr) bind(c, name="cudaGetDeviceProperties")
      import c_int, cudaDeviceProp
      type(cudaDeviceProp), intent(out) :: prop
      integer(c_int), intent(in), value :: device
      integer(c_int) :: ierr
    end function cudaGetDeviceProperties
    function cudaDeviceReset() &
        result(ierr) bind(c, name="cudaDeviceReset")
      import c_int
      integer(c_int) :: ierr
    end function cudaDeviceReset
    function cudaDeviceSynchronize() &
        result(ierr) bind(c, name="cudaDeviceSynchronize")
      import c_int
      integer(c_int) :: ierr
    end function cudaDeviceSynchronize
    function cudaGetLastError() &
        result(ierr) bind(c, name="cudaGetLastError")
      import c_int
      integer(c_int) :: ierr
    end function cudaGetLastError
    function cudaMemGetInfo(free, total) &
        result(ierr) bind(c, name="cudaMemGetInfo")
      import c_int, c_size_t
      integer(c_size_t) :: free, total
      integer(c_int) :: ierr
    end function cudaMemGetInfo
    function cudaDeviceGetLimit(pval, limit) &
        result(ierr) bind(c, name="cudaDeviceGetLimit")
      import c_int, c_size_t
      integer(c_size_t) :: pval
      integer(c_int), value :: limit
      integer(c_int) :: ierr
    end function cudaDeviceGetLimit
    function cudaDeviceSetLimit(limit, pval) &
        result(ierr) bind(c, name="cudaDeviceSetLimit")
      import c_int, c_size_t
      integer(c_int), value :: limit
      integer(c_size_t), value :: pval
      integer(c_int) :: ierr
    end function cudaDeviceSetLimit
  end interface

  !! CUDA Malloc
  public :: cudaMalloc
  public :: cudaFree

  interface
    function cudaMalloc(devPtr, size) &
        result(ierr) bind(c, name="cudaMalloc")
      import c_ptr, c_size_t, c_int
      type(c_ptr) :: devPtr
      integer(c_size_t), value :: size
      integer(c_int) :: ierr
    end function cudaMalloc
    function cudaFree(devPtr) &
        result(ierr) bind(c, name="cudaFree")
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
  public :: cudaMemcpyHostToHost, cudaMemcpyHostToDevice, &
      cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault

  public :: cudaMemcpy
  public :: cudaMemcpyAsync

  interface
    function cudaMemcpy(dst, src, count, kind) &
        result(ierr) bind(c, name="cudaMemcpy")
      import c_ptr, c_size_t, c_int
      type(c_ptr), value :: dst
      type(c_ptr), intent(in), value :: src
      integer(c_size_t), value :: count
      integer(c_int), value :: kind
      integer(c_int) :: ierr
    end function cudaMemcpy
    function cudaMemcpyAsync(dst, src, count, kind, stream) &
        result(ierr) bind(c, name="cudaMemcpyAsync")
      import c_ptr, c_size_t, c_int
      type(c_ptr), value :: dst
      type(c_ptr), intent(in), value :: src
      integer(c_size_t), value :: count
      integer(c_int), value :: kind
      type(c_ptr), value :: stream
      integer(c_int) :: ierr
    end function cudaMemcpyAsync
  end interface

  !! CUDA host register
  enum, bind(c)
    enumerator :: &
        cudaHostRegisterDefault  = 0, &
        cudaHostRegisterPortable = 1, &
        cudaHostRegisterMapped   = 2, &
        cudaHostRegisterIoMemory = 4
  end enum
  public :: cudaHostRegisterDefault, cudaHostRegisterPortable, &
      cudaHostRegisterMapped, cudaHostRegisterIoMemory

  public :: cudaHostRegister
  public :: cudaHostUnregister

  interface
    function cudaHostRegister(ptr, size, flags) &
        result(ierr) bind(c, name="cudaHostRegister")
      import c_ptr, c_int, c_size_t
      type(c_ptr), value :: ptr
      integer(c_size_t), value :: size
      integer(c_int), value :: flags
      integer(c_int) :: ierr
    end function cudaHostRegister
    function cudaHostUnregister(ptr) &
        result(ierr) bind(c, name="cudaHostUnregister")
      import c_ptr, c_int
      type(c_ptr), value :: ptr
      integer(c_int) :: ierr
    end function cudaHostUnregister
  end interface

  !! CUDA Streams
  public :: cudaStreamCreate
  public :: cudaStreamDestroy
  public :: cudaStreamSynchronize

  interface
    function cudaStreamCreate(pstream) &
        result(ierr) bind(c, name="cudaStreamCreate")
      import c_ptr, c_int
      type(c_ptr) :: pstream
      integer(c_int) :: ierr
    end function cudaStreamCreate
    function cudaStreamDestroy(pstream) &
        result(ierr) bind(c, name="cudaStreamDestroy")
      import c_ptr, c_int
      type(c_ptr), value :: pstream
      integer(c_int) :: ierr
    end function cudaStreamDestroy
    function cudaStreamSynchronize(pstream) &
        result(ierr) bind(c, name="cudaStreamSynchronize")
      import c_ptr, c_int
      type(c_ptr), value :: pstream
      integer(c_int) :: ierr
    end function cudaStreamSynchronize
  end interface

end module cuda_c_binding
