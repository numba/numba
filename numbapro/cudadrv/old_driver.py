'''
This is more-or-less a object-oriented interface to the CUDA driver API.
It properly has a lot of resemblence with PyCUDA.
'''
raise ImportError
import sys, os
import functools
import contextlib
from ctypes import (c_int, c_void_p, POINTER, c_size_t, byref, addressof,
                    c_uint, c_uint8, c_char_p, c_float, c_char,)
import ctypes
from .error import CudaDriverError, CudaSupportError
from numbapro._utils import mviewbuf
import threading
import numbapro


#------------------
# Configuration

VERBOSE_JIT_LOG = int(os.environ.get('NUMBAPRO_VERBOSE_CU_JIT_LOG', 1))
MIN_REQUIRED_CC = (2, 0)

# debug memory
debug_memory = False
debug_memory_alloc = 0
debug_memory_free = 0

def print_debug_memory():
    print "CUDA allocation: %d" % (debug_memory_alloc)
    print "CUDA free:       %d" % (debug_memory_free)


#------------------
# CUDA specific typedefs
cu_device = c_int
cu_device_attribute = c_int     # enum
cu_context = c_void_p           # an opaque handle
cu_module = c_void_p            # an opaque handle
cu_jit_option = c_int           # enum
cu_jit_input_type = c_int       # enum
cu_function = c_void_p          # an opaque handle
cu_device_ptr = c_size_t        # defined as unsigned int on 32-bit
                                # and unsigned long long on 64-bit machine
cu_stream = c_void_p            # an opaque handle
cu_event = c_void_p
cu_link_state = c_void_p

CUDA_SUCCESS                              = 0
CUDA_ERROR_INVALID_VALUE                  = 1
CUDA_ERROR_OUT_OF_MEMORY                  = 2
CUDA_ERROR_NOT_INITIALIZED                = 3
CUDA_ERROR_DEINITIALIZED                  = 4
CUDA_ERROR_PROFILER_DISABLED              = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8
CUDA_ERROR_NO_DEVICE                      = 100
CUDA_ERROR_INVALID_DEVICE                 = 101
CUDA_ERROR_INVALID_IMAGE                  = 200
CUDA_ERROR_INVALID_CONTEXT                = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202
CUDA_ERROR_MAP_FAILED                     = 205
CUDA_ERROR_UNMAP_FAILED                   = 206
CUDA_ERROR_ARRAY_IS_MAPPED                = 207
CUDA_ERROR_ALREADY_MAPPED                 = 208
CUDA_ERROR_NO_BINARY_FOR_GPU              = 209
CUDA_ERROR_ALREADY_ACQUIRED               = 210
CUDA_ERROR_NOT_MAPPED                     = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213
CUDA_ERROR_ECC_UNCORRECTABLE              = 214
CUDA_ERROR_UNSUPPORTED_LIMIT              = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216
CUDA_ERROR_INVALID_SOURCE                 = 300
CUDA_ERROR_FILE_NOT_FOUND                 = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303
CUDA_ERROR_OPERATING_SYSTEM               = 304
CUDA_ERROR_INVALID_HANDLE                 = 400
CUDA_ERROR_NOT_FOUND                      = 500
CUDA_ERROR_NOT_READY                      = 600
CUDA_ERROR_LAUNCH_FAILED                  = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701
CUDA_ERROR_LAUNCH_TIMEOUT                 = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709
CUDA_ERROR_ASSERT                         = 710
CUDA_ERROR_TOO_MANY_PEERS                 = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713
CUDA_ERROR_UNKNOWN                        = 999


# no preference for shared memory or L1 (default)
CU_FUNC_CACHE_PREFER_NONE    = 0x00
# prefer larger shared memory and smaller L1 cache
CU_FUNC_CACHE_PREFER_SHARED  = 0x01
# prefer larger L1 cache and smaller shared memory
CU_FUNC_CACHE_PREFER_L1      = 0x02
# prefer equal sized L1 cache and shared memory
CU_FUNC_CACHE_PREFER_EQUAL   = 0x03

# Automatic scheduling
CU_CTX_SCHED_AUTO          = 0x00
# Set spin as default scheduling
CU_CTX_SCHED_SPIN          = 0x01
# Set yield as default scheduling
CU_CTX_SCHED_YIELD         = 0x02
# Set blocking synchronization as default scheduling
CU_CTX_SCHED_BLOCKING_SYNC = 0x04

CU_CTX_SCHED_MASK          = 0x07

# Support mapped pinned allocations
CU_CTX_MAP_HOST            = 0x08
# Keep local memory allocation after launch
CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10

CU_CTX_FLAGS_MASK          = 0x1f



# If set, host memory is portable between CUDA contexts.
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_PORTABLE = 0x01

# If set, host memory is mapped into CUDA address space and
# cuMemHostGetDevicePointer() may be called on the host pointer.
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_DEVICEMAP = 0x02

# If set, host memory is allocated as write-combined - fast to write,
# faster to DMA, slow to read except via SSE4 streaming load instruction
# (MOVNTDQA).
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04

# If set, host memory is portable between CUDA contexts.
# Flag for cuMemHostRegister()
CU_MEMHOSTREGISTER_PORTABLE = 0x01

# If set, host memory is mapped into CUDA address space and
# cuMemHostGetDevicePointer() may be called on the host pointer.
# Flag for cuMemHostRegister()
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02


# Default event flag
CU_EVENT_DEFAULT        = 0x0
# Event uses blocking synchronization
CU_EVENT_BLOCKING_SYNC  = 0x1
# Event will not record timing data
CU_EVENT_DISABLE_TIMING = 0x2
# Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
CU_EVENT_INTERPROCESS   = 0x4

# The CUcontext on which a pointer was allocated or registered
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
# The CUmemorytype describing the physical location of a pointer
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
# The address at which a pointer's memory may be accessed on the device
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
# The address at which a pointer's memory may be accessed on the host
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
# A pair of tokens for use with the nv-p2p.h Linux kernel interface
CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5

# Host memory
CU_MEMORYTYPE_HOST    = 0x01
# Device memory
CU_MEMORYTYPE_DEVICE  = 0x02
# Array memory
CU_MEMORYTYPE_ARRAY   = 0x03
# Unified device or host memory
CU_MEMORYTYPE_UNIFIED = 0x04



# Compiled device-class-specific device code
# Applicable options: none
CU_JIT_INPUT_CUBIN = 0

# PTX source code
# Applicable options: PTX compiler options
CU_JIT_INPUT_PTX = 1

# Bundle of multiple cubins and/or PTX of some device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_FATBINAR = 2

# Host object with embedded device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_OBJECT = 3

# Archive of host objects with embedded device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_LIBRARY = 4



# Max number of registers that a thread may use.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_MAX_REGISTERS = 0,


# IN: Specifies minimum number of threads per block to target compilation
# for
# OUT: Returns the number of threads the compiler actually targeted.
# This restricts the resource utilization fo the compiler (e.g. max
# registers) such that a block with the given number of threads should be
# able to launch based on register limitations. Note, this option does not
# currently take into account any other resource limitations, such as
# shared memory utilization.
# Cannot be combined with ::CU_JIT_TARGET.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_THREADS_PER_BLOCK = 1


# Overwrites the option value with the total wall clock time, in
# milliseconds, spent in the compiler and linker
# Option type: float
# Applies to: compiler and linker

CU_JIT_WALL_TIME = 2


# Pointer to a buffer in which to print any log messages
# that are informational in nature (the buffer size is specified via
# option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
# Option type: char *
# Applies to: compiler and linker

CU_JIT_INFO_LOG_BUFFER = 3


# IN: Log buffer size in bytes.  Log messages will be capped at this size
# (including null terminator)
# OUT: Amount of log buffer filled with messages
# Option type: unsigned int
# Applies to: compiler and linker

CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4


# Pointer to a buffer in which to print any log messages that
# reflect errors (the buffer size is specified via option
# ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
# Option type: char *
# Applies to: compiler and linker

CU_JIT_ERROR_LOG_BUFFER = 5


# IN: Log buffer size in bytes.  Log messages will be capped at this size
# (including null terminator)
# OUT: Amount of log buffer filled with messages
# Option type: unsigned int
# Applies to: compiler and linker

CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6


# Level of optimizations to apply to generated code (0 - 4), with 4
# being the default and highest level of optimizations.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_OPTIMIZATION_LEVEL = 7


# No option value required. Determines the target based on the current
# attached context (default)
# Option type: No option value needed
# Applies to: compiler and linker

CU_JIT_TARGET_FROM_CUCONTEXT = 8


# Target is chosen based on supplied ::CUjit_target.  Cannot be
# combined with ::CU_JIT_THREADS_PER_BLOCK.
# Option type: unsigned int for enumerated type ::CUjit_target
# Applies to: compiler and linker

CU_JIT_TARGET = 9


# Specifies choice of fallback strategy if matching cubin is not found.
# Choice is based on supplied ::CUjit_fallback.
# Option type: unsigned int for enumerated type ::CUjit_fallback
# Applies to: compiler only

CU_JIT_FALLBACK_STRATEGY = 10


# Specifies whether to create debug information in output (-g)
# (0: false, default)
# Option type: int
# Applies to: compiler and linker

CU_JIT_GENERATE_DEBUG_INFO = 11


# Generate verbose log messages (0: false, default)
# Option type: int
# Applies to: compiler and linker

CU_JIT_LOG_VERBOSE = 12


# Generate line number information (-lineinfo) (0: false, default)
# Option type: int
# Applies to: compiler only

CU_JIT_GENERATE_LINE_INFO = 13


# Specifies whether to enable caching explicitly (-dlcm)
# Choice is based on supplied ::CUjit_cacheMode_enum.
# Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum
# Applies to: compiler only

CU_JIT_CACHE_MODE = 14


def _build_reverse_error_map():
    prefix = 'CUDA_ERROR'
    module = sys.modules[__name__]
    return dict((getattr(module, i), i)
                for i in filter(lambda x: x.startswith(prefix), globals()))

class ResourceManager(object):
    '''Per device resource manager
    '''
    def __init__(self, device_id):
        self.device_id = device_id
        self.reset()

    def reset(self):
        self.allocated = {}
        self.resources = {}
        self.pending_free = []
        self.pending_destroy = []
        self.can_release = True

    @property
    def driver(self):
        return Driver()

    #    @property
    #    def device(self):
    #        return Device(self.device_id)

    def add_memory(self, handle, dtor):
        assert handle not in self.allocated
        if debug_memory:
            global debug_memory_alloc
            debug_memory_alloc += 1
        self.allocated[handle] = dtor

    def free_memory(self, handle, later=False):
        dtor = self.allocated.pop(handle, None)
        if dtor:
            self.pending_free.append((handle, dtor))
            if not later:
                self.free_pending()

    def _free_memory(self, dtor, handle):
        if debug_memory:
            global debug_memory_free
            debug_memory_free += 1
        self.driver.check_error(dtor(handle), msg="Fail to free memory")

    def free_pending(self):
        if self.can_release:
            while self.pending_free:
                handle, dtor = self.pending_free.pop()
                self._free_memory(dtor, handle)
            while self.pending_destroy:
                handle, dtor, msg = self.pending_destroy.pop()
                self._destroy_resource(dtor, handle, msg)

    def _destroy_resource(self, dtor, handle, msg):
        self.driver.check_error(dtor(handle), msg=msg)

    def add_resource(self, handle, dtor):
        assert handle not in self.resources
        self.resources[handle] = dtor

    def free_resource(self, handle, msg, later=False):
        dtor = self.resources.pop(handle, None)
        if dtor:
            self.pending_destroy.append((handle, dtor, msg))
            if not later:
                self.free_pending()

class Driver(object):
    '''Facade to the CUDA Driver API.  A singleton class.  It is safe to
    construct new instance.  The constructor (__new__) will only create a new
    driver object if none exist already.
    '''

    '''
    Only the ones that we use are listed.
    '''
    API_PROTOTYPES = {
        # CUresult cuInit(unsigned int Flags);
        'cuInit' :              (c_int, c_uint),

        # CUresult cuDriverGetVersion ( int* driverVersion )
        'cuDriverGetVersion':   (c_int, POINTER(c_int)),

        # CUresult cuDeviceGetCount(int *count);
        'cuDeviceGetCount':     (c_int, POINTER(c_int)),

        # CUresult cuDeviceGet(CUdevice *device, int ordinal);
        'cuDeviceGet':          (c_int, POINTER(cu_device), c_int),

        # CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev )
        'cuDeviceGetName':      (c_int, c_char_p, c_int, cu_device),

        # CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
        #                               CUdevice dev);
        'cuDeviceGetAttribute': (c_int, POINTER(c_int), cu_device_attribute,
                                 cu_device),

        # CUresult cuDeviceComputeCapability(int *major, int *minor,
        #                                    CUdevice dev);
        'cuDeviceComputeCapability': (c_int, POINTER(c_int), POINTER(c_int),
                                      cu_device),

        # CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags,
        #                      CUdevice dev);
        'cuCtxCreate':          (c_int, POINTER(cu_context), c_uint, cu_device),

        # CUresult cuCtxGetDevice	(	CUdevice * 	device	 )
        'cuCtxGetDevice':       (c_int, POINTER(cu_device)),

        # CUresult cuCtxGetCurrent	(CUcontext *pctx);
        'cuCtxGetCurrent':      (c_int, POINTER(cu_context)),

        # CUresult cuCtxPopCurrent	(CUcontext *pctx);
        'cuCtxPopCurrent':      (c_int, POINTER(cu_context)),

        # CUresult cuCtxPushCurrent	(CUcontext pctx);
        'cuCtxPushCurrent':      (c_int, cu_context),

        # CUresult cuCtxDestroy(CUcontext pctx);
        'cuCtxDestroy':         (c_int, cu_context),

        # CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
        #                             unsigned int numOptions,
        #                             CUjit_option *options,
        #                             void **optionValues);
        'cuModuleLoadDataEx':   (c_int, cu_module, c_void_p, c_uint,
                                POINTER(cu_jit_option), POINTER(c_void_p)),

        # CUresult cuModuleUnload(CUmodule hmod);
        'cuModuleUnload':       (c_int, cu_module),

        # CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
        #                              const char *name);
        'cuModuleGetFunction':  (c_int, cu_function, cu_module, c_char_p),

        # CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc,
        #                                       CUfunc_cache config);
        'cuFuncSetCacheConfig': (c_int, cu_function, c_uint),

        # CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
        'cuMemAlloc':         (c_int, POINTER(cu_device_ptr), c_size_t),

        # CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
        'cuMemsetD8':         (c_int, cu_device_ptr, c_uint8, c_size_t),

        # CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
        #                          size_t N, CUstream hStream);
        'cuMemsetD8Async':    (c_int,
                               cu_device_ptr, c_uint8, c_size_t, cu_stream),

        # CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
        #                       size_t ByteCount);
        'cuMemcpyHtoD':         (c_int, cu_device_ptr, c_void_p, c_size_t),

        # CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
        #                            size_t ByteCount, CUstream hStream);
        'cuMemcpyHtoDAsync':    (c_int, cu_device_ptr, c_void_p, c_size_t,
                                cu_stream),

        # CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
        #                       size_t ByteCount);
        'cuMemcpyDtoD':         (c_int, cu_device_ptr, cu_device_ptr, c_size_t),

        # CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
        #                            size_t ByteCount, CUstream hStream);
        'cuMemcpyDtoDAsync':    (c_int, cu_device_ptr, cu_device_ptr, c_size_t,
                                cu_stream),


        # CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
        #                       size_t ByteCount);
        'cuMemcpyDtoH':         (c_int, c_void_p, cu_device_ptr, c_size_t),

        # CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
        #                            size_t ByteCount, CUstream hStream);
        'cuMemcpyDtoHAsync':    (c_int, c_void_p, cu_device_ptr, c_size_t,
                                cu_stream),

        # CUresult cuMemFree(CUdeviceptr dptr);
        'cuMemFree':            (c_int, cu_device_ptr),

        # CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
        'cuStreamCreate':       (c_int, POINTER(cu_stream), c_uint),

        # CUresult cuStreamDestroy(CUstream hStream);
        'cuStreamDestroy':      (c_int, cu_stream),

        # CUresult cuStreamSynchronize(CUstream hStream);
        'cuStreamSynchronize':  (c_int, cu_stream),

        # CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
        #                        unsigned int gridDimY,
        #                        unsigned int gridDimZ,
        #                        unsigned int blockDimX,
        #                        unsigned int blockDimY,
        #                        unsigned int blockDimZ,
        #                        unsigned int sharedMemBytes,
        #                        CUstream hStream, void **kernelParams,
        #                        void ** extra)
        'cuLaunchKernel':       (c_int, cu_function, c_uint, c_uint, c_uint,
                                 c_uint, c_uint, c_uint, c_uint, cu_stream,
                                 POINTER(c_void_p), POINTER(c_void_p)),

        #  CUresult cuMemHostAlloc	(	void ** 	pp,
        #                               size_t 	bytesize,
        #                               unsigned int 	Flags
        #                           )
        'cuMemHostAlloc': (c_int, c_void_p, c_size_t, c_uint),

        #  CUresult cuMemFreeHost	(	void * 	p	 )
        'cuMemFreeHost': (c_int, c_void_p),

        # CUresult cuMemHostRegister(void * 	p,
        #                            size_t 	bytesize,
        #                            unsigned int 	Flags)
        'cuMemHostRegister':    (c_int, c_void_p, c_size_t, c_uint),

        # CUresult cuMemHostUnregister(void * 	p)
        'cuMemHostUnregister':  (c_int, c_void_p),

        # CUresult cuMemHostGetDevicePointer(CUdeviceptr * pdptr,
        #                                    void *        p,
        #                                    unsigned int  Flags)
        'cuMemHostGetDevicePointer': (c_int, POINTER(cu_device_ptr),
                                      c_void_p, c_uint),

        # CUresult cuMemGetInfo(size_t * free, size_t * total)
        'cuMemGetInfo' : (c_int, POINTER(c_size_t), POINTER(c_size_t)),

        # CUresult cuEventCreate	(	CUevent * 	phEvent,
        #                               unsigned int 	Flags )
        'cuEventCreate': (c_int, POINTER(cu_event), c_uint),

        # CUresult cuEventDestroy	(	CUevent 	hEvent	 )
        'cuEventDestroy': (c_int, cu_event),

        # CUresult cuEventElapsedTime	(	float * 	pMilliseconds,
        #                                   CUevent 	hStart,
        #                                   CUevent 	hEnd )
        'cuEventElapsedTime': (c_int, POINTER(c_float), cu_event, cu_event),

        # CUresult cuEventQuery	(	CUevent 	hEvent	 )
        'cuEventQuery': (c_int, cu_event),

        # CUresult cuEventRecord	(	CUevent 	hEvent,
        #                               CUstream 	hStream )
        'cuEventRecord': (c_int, cu_event, cu_stream),

        # CUresult cuEventSynchronize	(	CUevent 	hEvent	 )
        'cuEventSynchronize': (c_int, cu_event),


        # CUresult cuStreamWaitEvent	(	CUstream        hStream,
        #                                   CUevent         hEvent,
        #                                	unsigned int 	Flags )
        'cuStreamWaitEvent': (c_int, cu_stream, cu_event, c_uint),

        # CUresult 	cuPointerGetAttribute (void *data, CUpointer_attribute attribute, CUdeviceptr ptr)
        'cuPointerGetAttribute': (c_int, c_void_p, c_uint, cu_device_ptr),

        #    CUresult cuMemGetAddressRange	(	CUdeviceptr * 	pbase,
        #                                        size_t * 	psize,
        #                                        CUdeviceptr 	dptr
        #                                        )
        'cuMemGetAddressRange': (c_int,
                                 POINTER(cu_device_ptr),
                                 POINTER(c_size_t),
                                 cu_device_ptr),

        #    CUresult cuMemHostGetFlags	(	unsigned int * 	pFlags,
        #                                   void * 	p )
        'cuMemHostGetFlags': (c_int,
                              POINTER(c_uint),
                              c_void_p),

        #   CUresult cuCtxSynchronize ( void )
        'cuCtxSynchronize' : (c_int,),

        #    CUresult
        #    cuLinkCreate(unsigned int numOptions, CUjit_option *options,
        #                 void **optionValues, CUlinkState *stateOut);
        'cuLinkCreate': (c_int,
                         c_uint, POINTER(cu_jit_option),
                         POINTER(c_void_p), POINTER(cu_link_state)),

        #    CUresult
        #    cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
        #                  size_t size, const char *name, unsigned
        #                  int numOptions, CUjit_option *options,
        #                  void **optionValues);
        'cuLinkAddData': (c_int,
                          cu_link_state, cu_jit_input_type, c_void_p,
                          c_size_t, c_char_p, c_uint, POINTER(cu_jit_option),
                          POINTER(c_void_p)),

        #    CUresult
        #    cuLinkAddFile(CUlinkState state, CUjitInputType type,
        #                  const char *path, unsigned int numOptions,
        #                  CUjit_option *options, void **optionValues);

        'cuLinkAddFile': (c_int,
                          cu_link_state, cu_jit_input_type, c_char_p, c_uint,
                          POINTER(cu_jit_option), POINTER(c_void_p)),

        #    CUresult CUDAAPI
        #    cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut)
        'cuLinkComplete': (c_int,
                           cu_link_state, POINTER(c_void_p), POINTER(c_size_t)),

        #    CUresult CUDAAPI
        #    cuLinkDestroy(CUlinkState state)
        'cuLinkDestroy': (c_int, cu_link_state),


        # NOTE: this is a dummy function to test if the defer error reporting
        'easteregg': (c_int,),
    }

    _REVERSE_ERROR_MAP = _build_reverse_error_map()

    __INSTANCE = None

    # A mapping from context handle -> Context instance
    _CONTEXTS = {}

    # Thread local storage for cache the context
    _THREAD_LOCAL = threading.local()

    def __new__(cls, override_path=None):
        if cls.__INSTANCE is None:
            # Prevent driver loading at `import numbapro`
            assert numbapro._initialization_completed

            # Actual driver initialization
            cls.__INSTANCE = inst = object.__new__(Driver)

            # Determine DLL type
            if sys.platform == 'win32':
                dlloader = ctypes.WinDLL
                dldir = ['\\windows\\system32']
                dlname = 'nvcuda.dll'
            elif sys.platform == 'darwin':
                dlloader = ctypes.CDLL
                dldir = ['/usr/local/cuda/lib']
                dlname = 'libcuda.dylib'
            else:
                dlloader = ctypes.CDLL
                dldir = ['/usr/lib', '/usr/lib64']
                dlname = 'libcuda.so'

            # First search for the name in the default library path.
            # If that is not found, try the specific path.
            candidates = [dlname] + [os.path.join(x, dlname) for x in dldir]

            if override_path:
                # If override_path is provided, use it and ignore the others
                candidates = [override_path]
            else:
                envpath = os.environ.get('NUMBAPRO_CUDA_DRIVER')
                if envpath:
                    # If envvar is provided, use it and ignore the others
                    candidates = [envpath]

            # Load the driver
            path_not_exist = []
            driver_load_error = []
            for path in candidates:
                try:
                    inst.driver = dlloader(path)
                    inst.path = path
                except OSError as e:
                    # Problem opening the DLL
                    path_not_exist.append(not os.path.isfile(path))
                    driver_load_error.append(e)
                    pass
                else:
                    break       # got it; break out
            else:
                # not found, barf
                if all(path_not_exist):
                    cls._raise_driver_not_found()
                else:
                    errmsg = '\n'.join(str(e) for e in driver_load_error)
                    cls._raise_driver_error(errmsg)

            # Obtain function pointers
            def make_poison(func):
                def poison(*args, **kws):
                    msg = ("incompatible CUDA driver: "
                           "function %s not found. "
                           "\nrequires CUDA driver distributed with CUDA 5.5")
                    raise CudaDriverError(msg % (func,))
                return poison

            for func, prototype in inst.API_PROTOTYPES.items():
                restype = prototype[0]
                argtypes = prototype[1:]
                try:
                    ct_func = inst._cu_symbol_newer(func)
                except AttributeError:
                    ct_func = make_poison(func)
                else:
                    ct_func.restype = restype
                    ct_func.argtypes = argtypes

                setattr(inst, func, ct_func)

            # initialize the API
            try:
                error = inst.cuInit(0)
                inst.check_error(error, "Failed to initialize CUDA driver")
            except AttributeError as e:
                # Not a real driver?
                cls._raise_driver_error(e)
            except CudaDriverError as e:
                # it doesn't work?
                cls._raise_driver_error(e)

        return cls.__INSTANCE

    @property
    def version(self):
        ver = c_int(0)
        self.cuDriverGetVersion(byref(ver))
        return ver.value

    @classmethod
    def _raise_driver_not_found(cls):
        cls.__INSTANCE = None # posion
        raise CudaSupportError(
                   "CUDA is not supported or the library cannot be found. "
                   "Try setting environment variable NUMBAPRO_CUDA_DRIVER "
                   "with the path of the CUDA driver shared library.")

    @classmethod
    def _raise_driver_error(cls, e):
        cls.__INSTANCE = None # posion
        raise CudaSupportError("Error loading CUDA library:\n%s" % e)

    def _cu_symbol_newer(self, symbol):
        try:
            return getattr(self.driver, '%s_v2' % symbol)
        except AttributeError:
            return getattr(self.driver, symbol)

    def get_device_count(self):
        count = c_int()
        error = self.cuDeviceGetCount(byref(count))
        self.check_error(error, 'Failed to get number of device')
        return count.value

    def check_error(self, error, msg):
        if error:
            exc = CudaDriverError('%s\n%s\n' %
                                  (self._REVERSE_ERROR_MAP[error], msg))
            if error == CUDA_ERROR_LAUNCH_FAILED:
                ctx = self.get_context_tls()
                if ctx:
                    ctx.device.reset()
            raise exc

    def create_context(self, device=None):
        '''Create a new context.

        NOTE: If there is already a context for this module,
              this function will raise Exception.
              We do not support multiple contexts per thread, yet.

        device --- [optional] The device object to be used for the new context.
        '''
        if device is None:
            device = Device(0)
        if self.current_context(noraise=True) is not None:
            errmsg = "Does not support multiple context per thread, yet."
            raise Exception(errmsg)
        ctxt = _Context(device)
        self._CONTEXTS[ctxt._handle.value] = ctxt
        self._cache_current_context(ctxt)
        return ctxt

    def get_context_tls(self):
        try:
            handle = self._THREAD_LOCAL.context
        except AttributeError:
            return None
        else:
            return self._CONTEXTS[handle]

    def current_context(self, noraise=False):
        '''Get current context from TLS
        '''
        ctxt = self.get_context_tls()
        if ctxt is None:
            ctxt = self._get_current_context(noraise=noraise)
            if ctxt:
                self._cache_current_context(ctxt)
        return ctxt

    def _cache_current_context(self, ctxt):
        '''Store current context into TLS
        '''
        self._THREAD_LOCAL.context = ctxt._handle.value

    def _get_current_context(self, noraise=False):
        '''Uses CUDA driver API to get current context
        '''
        handle = cu_context()
        error = self.cuCtxGetCurrent(byref(handle))
        self.check_error(error, "Fail to get current context.")
        if handle.value is None:
            if noraise:
                return None
            else:
                raise CudaDriverError("No CUDA context was created.")
        elif handle.value in self._CONTEXTS:
            context = self._CONTEXTS[handle.value]
            return context
        else:
            # context is created externally
            context = _Context(handle=handle)
            self._CONTEXTS[handle.value] = context
            self._cache_current_context(context)
            return context

    def get_or_create_context(self, device=None):
        '''Returns the current context if exists, or get create a new one.
        '''
        try:
            return self.current_context()
        except CudaDriverError:
            return self.create_context(device)

    def release_context(self, context):
        rm = context.device.resource_manager
        rm.free_pending()
        handle = context._handle.value
        del self._CONTEXTS[handle]
        if handle == self._THREAD_LOCAL.context:
            # Remove all thread local
            for k in vars(self._THREAD_LOCAL).keys():
                delattr(self._THREAD_LOCAL, k)

    def reset_context(self):
        ctx = self.get_context_tls()
        device = ctx.device
        self.release_context(ctx)
        # discard current context
        curctx = cu_context(0)
        self.cuCtxPopCurrent(byref(curctx))
        # create new context
        self.create_context(device)


CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34

class Device(object):

    ATTRIBUTES = { # attributes with integer values
      'MAX_THREADS_PER_BLOCK': CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      'MAX_GRID_DIM_X':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      'MAX_GRID_DIM_Y':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      'MAX_GRID_DIM_Z':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      'MAX_BLOCK_DIM_X':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
      'MAX_BLOCK_DIM_Y':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
      'MAX_BLOCK_DIM_Z':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      'MAX_SHARED_MEMORY':     CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
      'ASYNC_ENGINE_COUNT':    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
      'CAN_MAP_HOST_MEMORY':   CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
      'MULTIPROCESSOR_COUNT':  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
      'WARP_SIZE':             CU_DEVICE_ATTRIBUTE_WARP_SIZE,
      'UNIFIED_ADDRESSING':    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
      'PCI_BUS_ID':            CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
      'PCI_DEVICE_ID':         CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
    }

    _resources = {}   # mapping from device to resource manager

    def __init__(self, device_id):
        self.driver = Driver()
        got_device = c_int()
        error = self.driver.cuDeviceGet(byref(got_device), device_id)
        self.driver.check_error(error, 'Failed to get device %d' % device_id)
        assert device_id == got_device.value, "driver returned another device"
        self.id = got_device.value
        self.__read_name()
        self.__read_attributes()

        self.resource_manager = ResourceManager(device_id)
        self._resources[device_id] = self.resource_manager

    def reset(self):
        self.resource_manager.reset()
        self.driver.reset_context()

    def __repr__(self):
        return "<CUDA device %d '%s'>" % (self.id, self.name)

    def get_memory_info(self):
        "Returns (free, total) memory in bytes on the device."
        free = c_size_t()
        total = c_size_t()
        error = self.driver.cuMemGetInfo(byref(free), byref(total))
        msg = 'Failed to get memory info on device %d' % self.id
        self.driver.check_error(error, msg)
        return free.value, total.value

    def __read_name(self):
        bufsz = 128
        buf = (c_char * bufsz)()
        error = self.driver.cuDeviceGetName(buf, bufsz, self.id)
        msg = 'Failed to get name of device %d' % self.id
        self.driver.check_error(error, msg)
        self.name = buf.value

    def __read_attributes(self):
        got_value = c_int()
        for name, num in self.ATTRIBUTES.items():
            error = self.driver.cuDeviceGetAttribute(byref(got_value), num,
                                                     self.id)
            msg = 'Failed to read attribute "%s" from %s' % (name, self)
            self.driver.check_error(error, msg)
            setattr(self, name, got_value.value)

        got_major = c_int()
        got_minor = c_int()
        error = self.driver.cuDeviceComputeCapability(byref(got_major),
                                                      byref(got_minor),
                                                      self.id)
        msg = 'Failed to read compute capability from %s' % self
        self.driver.check_error(error, msg)

        setattr(self, 'COMPUTE_CAPABILITY', (got_major.value, got_minor.value))

    @property
    def attributes(self):
        '''Returns all attributes as a dictionary
        '''
        keys = list(self.ATTRIBUTES.keys())
        keys += ['COMPUTE_CAPABILITY']
        return dict((k, getattr(self, k)) for k in keys)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, rhs):
        try:
            return self.id == rhs.id
        except AttributeError:
            return False

    def __ne__(self, rhs):
        return not (self == rhs)


class _Context(object):
    def __init__(self, device=None, handle=None):
        if device is not None:
            assert handle is None
            self.device = device
            if self.device.COMPUTE_CAPABILITY < MIN_REQUIRED_CC:
                msg = ("failed to initialize %s\n"
                       "only support device with compute capability >=2.0\n"
                       "please use numbapro.check_cuda() to scan for supported "
                       "CUDA GPUs" % self.device)
                raise CudaSupportError(msg)
            self._handle = cu_context()
            flags = 0
            if self.device.CAN_MAP_HOST_MEMORY:
                flags |= CU_CTX_MAP_HOST
            else:
                # XXX: do I really need this?
                assert False, "unreachable: cannot map host memory"
            error = self.driver.cuCtxCreate(byref(self._handle), flags,
                                            self.device.id)
            self.driver.check_error(error, ('Failed to create context on %s' %
                                            self.device))
            self.device.resource_manager.add_resource(self._handle.value,
                                                      self.driver.cuCtxDestroy)
        else:
            # Do not register the context to resource manager
            assert handle is not None
            self._handle = handle
            device = cu_device()
            error = self.driver.cuCtxGetDevice(byref(device))
            self.driver.check_error(error, 'Failed to get device from context')
            self.device = Device(device.value)

    def __del__(self):
        self.device.resource_manager.free_resource(self._handle.value,
                                               msg='Failed to destroy context',
                                               later=False)

    @property
    def driver(self):
        return Driver()

    def synchronize(self):
        error = self.driver.cuCtxSynchronize()
        self.driver.check_error(error, 'Failed to synchronize context')

    def __str__(self):
        return 'Context %s on %s' % (id(self), self.device)


class Stream(object):
    def __init__(self):
        self.device = self.driver.current_context().device
        self._handle = cu_stream()

        flush_pending_free()
        error = self.driver.cuStreamCreate(byref(self._handle), 0)
        msg = 'Failed to create stream on %s' % self.device
        self.driver.check_error(error, msg)

        self.device.resource_manager.add_resource(self._handle.value,
                                                  self.driver.cuStreamDestroy)

    def __del__(self):
        self.device.resource_manager.free_resource(self._handle.value,
                                               msg='Failed to destroy stream',
                                               later=True)

    def __str__(self):
        return 'Stream %d on %s' % (self, self.device)

    def __int__(self):
        return self._handle.value

    def synchronize(self):
        '''Ensure all commands in the stream has been completed
        '''
        self.driver.cuStreamSynchronize(self._handle)

    @contextlib.contextmanager
    def auto_synchronize(self):
        '''Use this for create a context that synchronize automatically.
        '''
        yield self
        self.synchronize()

    @property
    def driver(self):
        return Driver()


class HostAllocMemory(mviewbuf.MemAlloc):
    '''A host allocation by the CUDA driver that is pagelocked.
    This object exposes the buffer interface; thus, user can use
    this as a buffer for ndarray.
    '''
    __cuda_memory__ = True
    def __init__(self, bytesize, mapped=False, portable=False, wc=False):
        self.device = self.driver.current_context().device
        self.bytesize = bytesize
        self._handle = c_void_p(0)
        flags = 0
        if mapped:
            flags |= CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= CU_MEMHOSTALLOC_WRITECOMBINED

        flush_pending_free()
        error = self.driver.cuMemHostAlloc(byref(self._handle), bytesize, flags)
        self.driver.check_error(error, 'Failed to host alloc')
        self.device.resource_manager.add_memory(self._handle.value,
                                                self.driver.cuMemFreeHost)

        self.devmem = cu_device_ptr(0)

        self.__cuda_memory__ = mapped
        if mapped:
            # get device pointer
            error = self.driver.cuMemHostGetDevicePointer(byref(self.devmem),
                                                          self._handle, 0)
            self.driver.check_error(error,
                                    'Failed to get device ptr from host ptr')

        self._buflen_ = self.bytesize
        self._bufptr_ = self._handle.value

    def __del__(self):
        self.device.resource_manager.free_memory(self._handle.value, later=True)

    @property
    def device_ctypes_pointer(self):
        return self.devmem

    @property
    def driver(self):
        return Driver()


class DeviceMemory(object):
    __cuda_memory__ = True

    def __init__(self, bytesize):
        self.device = self.driver.current_context().device
        self._allocate(bytesize)

    def __del__(self):
        self.device.resource_manager.free_memory(self._handle.value, later=True)

    def _allocate(self, bytesize):
        assert not hasattr(self, '_handle'), "_handle is already defined"
        self._handle = cu_device_ptr()

        flush_pending_free()
        error = self.driver.cuMemAlloc(byref(self._handle), bytesize)
        self.driver.check_error(error, 'Failed to allocate memory')
        self.device.resource_manager.add_memory(self._handle.value,
                                                self.driver.cuMemFree)
        self.bytesize = bytesize

    @property
    def device_ctypes_pointer(self):
        return self._handle

    @property
    def driver(self):
        return Driver()

class DeviceView(object):
    __cuda_memory__ = True
    def __init__(self, owner, start, stop=None):
        self.device = self.driver.current_context().device
        self._handle = cu_device_ptr(device_pointer(owner) + start)
        self._owner = owner
        if stop is None:
            sz = device_memory_size(owner) - start
        else:
            sz = stop - start
        assert sz > 0, "zero memory size"
        self._cuda_memsize_ = sz

    @property
    def device_ctypes_pointer(self):
        return self._handle

    @property
    def driver(self):
        return Driver()

class PinnedMemory(object):
    def __init__(self, owner, ptr, size, mapped=False):
        self._owner = owner
        self.device = self.driver.current_context().device
        if mapped and not self.device.CAN_MAP_HOST_MEMORY:
            raise CudaDriverError("Device %s cannot map host memory" %
                                  self.device)

        self._pointer = ptr
        # possible flags are "portable" (between context)
        # and "device-map" (map host memory to device thus no need
        # for memory transfer).
        flags = 0
        self._mapped = mapped
        if mapped:
            flags |= CU_MEMHOSTREGISTER_DEVICEMAP

        flush_pending_free()
        error = self.driver.cuMemHostRegister(ptr, size, flags)
        self.driver.check_error(error, 'Failed to pin memory')

        if mapped:
            self._get_device_pointer()

        self.__cuda_memory__ = mapped

        self.device.resource_manager.add_resource(self._pointer,
                                                self.driver.cuMemHostUnregister)

    def _get_device_pointer(self):
        assert self._mapped, "memory not mapped"
        self._devmem = cu_device_ptr(0)
        flags = 0 # must be zero for now
        error = self.driver.cuMemHostGetDevicePointer(byref(self._devmem),
                                                      self._pointer, flags)
        self.driver.check_error(error, 'Failed to get device ptr from host ptr')

    @property
    def device_ctypes_pointer(self):
        return self._devmem

    def __del__(self):
        self.device.resource_manager.free_resource(self._pointer,
                                               msg='Failed to unpin memory',
                                               later=True)

    @property
    def driver(self):
        return Driver()

class Module(object):
    def __init__(self, ptx=None, image=None):
        assert ptx is not None or image is not None, "internal error"
        if ptx is not None:
            self.load_data(c_char_p(ptx))
        elif image is not None:
            self.load_data(image)
        else:
            raise ValueError

    def load_data(self, image):
        """
        image must be a pointer
        """
        logsz = os.environ.get('NUMBAPRO_CUDA_LOG_SIZE', 1024)

        jitinfo = (c_char * logsz)()
        jiterrors = (c_char * logsz)()

        options = {
            CU_JIT_INFO_LOG_BUFFER              : addressof(jitinfo),
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES   : c_void_p(logsz),
            CU_JIT_ERROR_LOG_BUFFER             : addressof(jiterrors),
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  : c_void_p(logsz),
            CU_JIT_LOG_VERBOSE                  : c_void_p(VERBOSE_JIT_LOG),
        }

        option_keys = (cu_jit_option * len(options))(*options.keys())
        option_vals = (c_void_p * len(options))(*options.values())

        self._handle = cu_module()

        flush_pending_free()
        status = self.driver.cuModuleLoadDataEx(byref(self._handle),
                                                image,
                                                len(options),
                                                option_keys,
                                                option_vals)

        self.driver.check_error(status,
                                'Failed to load module:\n%s' % jiterrors.value)

        self.device = self.driver.current_context().device
        self.device.resource_manager.add_resource(self._handle.value,
                                                  self.driver.cuModuleUnload)
        self.info_log = jitinfo.value

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error =  driver.cuModuleUnload(handle)
        driver.check_error(error, 'Failed to unload module')

    def __del__(self):
        self.device.resource_manager.free_resource(self._handle.value,
                                               msg='Failed to unload module',
                                               later=True)

    @property
    def driver(self):
        return Driver()

class Function(object):

    griddim = 1, 1, 1
    blockdim = 1, 1, 1
    stream = None
    sharedmem = 0

    def __init__(self, module, name):
        self.device = self.driver.current_context().device
        self.module = module
        self.name = name
        self._handle = cu_function()
        error = self.driver.cuModuleGetFunction(byref(self._handle),
                                                self.module._handle,
                                                name);
        self.driver.check_error(error, 'Failed to get function "%s" from module' % name)


    @property
    def driver(self):
        return Driver()

    @property
    def context(self):
        return self.module.context

    def __str__(self):
        return 'CUDA kernel %s on %s' % (self.name, self)

    def cache_config(self, prefer_equal=False, prefer_cache=False, prefer_shared=False):
        prefer_equal = prefer_equal or (prefer_cache and prefer_shared)
        if prefer_equal:
            flag = CU_FUNC_CACHE_PREFER_EQUAL
        elif prefer_cache:
            flag = CU_FUNC_CACHE_PREFER_L1
        elif prefer_shared:
            flag = CU_FUNC_CACHE_PREFER_SHARED
        else:
            flag = CU_FUNC_CACHE_PREFER_NONE

        err = self.driver.cuFuncSetCacheConfig(self._handle, flag)
        self.driver.check_error(err, 'Failed to set cache config')

    def configure(self, griddim, blockdim, sharedmem=0, stream=None):
        import copy

        while len(griddim) < 3:
            griddim += (1,)

        while len(blockdim) < 3:
            blockdim += (1,)

        inst = copy.copy(self) # shallow clone the object
        inst.griddim = griddim
        inst.blockdim = blockdim
        inst.sharedmem = sharedmem
        if stream:
            inst.stream = stream._handle
        else:
            inst.stream = None
        return inst

    def __call__(self, *args):
        '''
        *args -- Must be either ctype objects of DevicePointer instances.
        '''
        assert self.driver.current_context().device is self.device, \
                "function not defined in current active context"
        launch_kernel(self._handle, self.griddim, self.blockdim,
                      self.sharedmem, self.stream, args)

class Event(object):
    def __init__(self, timing=True):
        self.device = self.driver.current_context().device
        self._handle = cu_event()
        self.timing = timing
        flags = 0
        if not timing:
            flags |= CU_EVENT_DISABLE_TIMING
        error = self.driver.cuEventCreate(byref(self._handle), flags)
        self.driver.check_error(error, 'Failed to create event')
        self.device.resource_manager.add_resource(self._handle.value,
                                                  self.driver.cuEventDestroy)

    def __del__(self):
        self.device.resource_manager.free_resource(self._handle.value,
                                               msg='Failed to destroy event',
                                               later=True)

    def query(self):
        '''Returns True if all work before the most recent record has completed;
        otherwise, returns False.
        '''
        status = self.driver.cuEventQuery(self._handle)
        if status == CUDA_ERROR_NOT_READY:
            return False
        self.driver.check_error(status, 'Failed to query event')
        return True

    def record(self, stream=0):
        '''Set the record state of the event at the stream.
        '''
        hstream = stream._handle if stream else 0
        error = self.driver.cuEventRecord(self._handle, hstream)
        self.driver.check_error(error, 'Failed to record event')

    def synchronize(self):
        '''Synchronize the host thread for the completion of the event.
        '''
        error = self.driver.cuEventSynchronize(self._handle)
        self.driver.check_error(error, 'Failed to synchronize event')

    def wait(self, stream=0):
        '''All future works submitted to stream will wait util the event
        completes.
        '''
        hstream = stream._handle if stream else 0
        flags = 0
        error = self.driver.cuStreamWaitEvent(hstream, self._handle, flags)
        self.driver.check_error(error, 'Failed to do stream wait event')

    def elapsed_time(self, evtend):
        return event_elapsed_time(self, evtend)

    @property
    def driver(self):
        return Driver()

FILE_EXTENSION_MAP = {
    'o'      : CU_JIT_INPUT_OBJECT,
    'ptx'    : CU_JIT_INPUT_PTX,
    'a'      : CU_JIT_INPUT_LIBRARY,
    'cubin'  : CU_JIT_INPUT_CUBIN,
    'fatbin' : CU_JIT_INPUT_FATBINAR,
}

class Linker(object):
    def __init__(self):
        self.driver = Driver()

        logsz = int(os.environ.get('NUMBAPRO_CUDA_LOG_SIZE', 1024))
        linkerinfo = (c_char * logsz)()
        linkererrors = (c_char * logsz)()

        options = {
            CU_JIT_INFO_LOG_BUFFER              : addressof(linkerinfo),
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES   : c_void_p(logsz),
            CU_JIT_ERROR_LOG_BUFFER             : addressof(linkererrors),
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  : c_void_p(logsz),
            CU_JIT_LOG_VERBOSE                  : c_void_p(1),
        }

        option_keys = (cu_jit_option * len(options))(*options.keys())
        option_vals = (c_void_p * len(options))(*options.values())

        self._handle = cu_link_state()
        status = self.driver.cuLinkCreate(len(options),
                                          option_keys,
                                          option_vals,
                                          byref(self._handle))

        self.driver.check_error(status, 'Failed to initialize linker')

        self.device = self.driver.current_context().device
        self.device.resource_manager.add_resource(self._handle.value,
                                                  self.driver.cuLinkDestroy)

        self.linker_info_buf = linkerinfo
        self.linker_errors_buf = linkererrors

        self._keep_alive = [linkerinfo, linkererrors, option_keys, option_vals]

    @property
    def info_log(self):
        return self.linker_info_buf.value

    @property
    def error_log(self):
        return self.linker_errors_buf.value

    def __del__(self):
        self.device.resource_manager.free_resource(self._handle.value,
                                               msg='Failed to destroy linker',
                                               later=True)

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        ptxbuf = c_char_p(ptx)
        namebuf = c_char_p(name)
        self._keep_alive += [ptxbuf, namebuf]

        status = self.driver.cuLinkAddData(self._handle,
                                           CU_JIT_INPUT_PTX,
                                           ptxbuf,
                                           len(ptx),
                                           namebuf,
                                           0, None, None)
        self.driver.check_error(status,
                                'Failed to add ptx: %s' % self.error_log)

    def add_file(self, path, kind):
        pathbuf = c_char_p(path)
        self._keep_alive.append(pathbuf)

        status = self.driver.cuLinkAddFile(self._handle,
                                           kind,
                                           pathbuf,
                                           0, None, None)
        self.driver.check_error(status, 'Failed to add file %s: %s' %
                                (path, self.error_log))


    def add_file_guess_ext(self, path):
        ext = path.rsplit('.', 1)[1]
        kind = FILE_EXTENSION_MAP[ext]
        self.add_file(path, kind)

    def complete(self):
        '''
        Returns (cubin, size)
            cubin is a pointer to a internal buffer of cubin owned
            by the linker; thus, it should be loaded before the linker
            is destroyed.
        '''
        cubin = c_void_p(0)
        size = c_size_t(0)
        status = self.driver.cuLinkComplete(self._handle,
                                            byref(cubin),
                                            byref(size))
        self.driver.check_error(status, 'Failed to link: %s' % self.error_log)
        assert size > 0, 'linker returned a zero sized cubin'
        del self._keep_alive[:]
        return cubin, size

def event_elapsed_time(evtstart, evtend):
    driver = evtstart.driver
    msec = c_float()
    err = driver.cuEventElapsedTime(byref(msec), evtstart._handle,
                                    evtend._handle)
    driver.check_error(err, 'Failed to get elapsed time')
    return msec.value

def launch_kernel(cufunc_handle, griddim, blockdim, sharedmem, hstream, args):
    driver = Driver()
    gx, gy, gz = griddim
    bx, by, bz = blockdim

    param_vals = []
    for arg in args:
        if is_device_memory(arg):
            param_vals.append(addressof(device_ctypes_pointer(arg)))
        else:
            param_vals.append(addressof(arg))

    params = (c_void_p * len(param_vals))(*param_vals)

    error = driver.cuLaunchKernel(
                cufunc_handle,
                gx, gy, gz,
                bx, by, bz,
                sharedmem,
                hstream,
                params,
                None)

    driver.check_error(error, "Failed to launch kernel")

def get_or_create_context():
    "Get the current context if it exists or create one."
    drv = Driver()
    context = drv.current_context(noraise=True)
    if not context:
        context = drv.create_context(Device(0))
    return context

def flush_pending_free():
    get_or_create_context().device.resource_manager.free_pending()

def require_context(fn):
    "Decorator to ensure a context exists for the current thread."
    @functools.wraps(fn)
    def _wrapper(*args, **kws):
        get_or_create_context()
        return fn(*args, **kws)
    return _wrapper

#
# CUDA memory functions
#

def _device_pointer_attr(devmem, attr, odata):
    """Query attribute on the device pointer
    """
    driver = Driver()
    attr = CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    error = driver.cuPointerGetAttribute(byref(odata), attr,
                                         device_ctypes_pointer(devmem))
    driver.check_error(error, "Failed to query pointer attribute")

def device_pointer_type(devmem):
    """Query the device pointer type: host, device, array, unified?
    """
    ptrtype = c_int(0)
    _device_pointer_attr(devmem, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptrtype)
    map = {
        CU_MEMORYTYPE_HOST    : 'host',
        CU_MEMORYTYPE_DEVICE  : 'device',
        CU_MEMORYTYPE_ARRAY   : 'array',
        CU_MEMORYTYPE_UNIFIED : 'unified',
    }
    return map[ptrtype.value]

def device_extents(devmem):
    """Find the extents (half open begin and end pointer) of the underlying
    device memory allocation.

    NOTE: it always returns the extents of the allocation but the extents
    of the device memory view that can be a subsection of the entire allocation.
    """
    driver = Driver()
    s = cu_device_ptr()
    n = c_size_t()
    devptr = device_ctypes_pointer(devmem)
    driver.cuMemGetAddressRange(byref(s), byref(n), devptr)
    s, n = s.value, n.value
    return s, s + n

def device_memory_size(devmem):
    """Check the memory size of the device memory.
    The result is cached in the device memory object.
    It may query the driver for the memory size of the device memory allocation.
    """
    sz = getattr(devmem, '_cuda_memsize_', None)
    if sz is None:
        s, e = device_extents(devmem)
        sz = e - s
        devmem._cuda_memsize_ = sz
    assert sz > 0, "zero length array"
    return sz

def host_pointer(obj):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    if isinstance(obj, (int, long)):
        return obj
    return mviewbuf.memoryview_get_buffer(obj)

def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    return mviewbuf.memoryview_get_extents(obj)

def memory_size_from_info(shape, strides, itemsize):
    """et the byte size of a contiguous memory buffer given the shape, strides
    and itemsize.
    """
    assert len(shape) == len(strides), "# dim mismatch"
    ndim = len(shape)
    s, e = mviewbuf.memoryview_get_extents_info(shape, strides, ndim, itemsize)
    return e - s

def host_memory_size(obj):
    "Get the size of the memory"
    s, e = host_memory_extents(obj)
    assert e >= s, "memory extend of negative size"
    return e - s

def device_pointer(obj):
    "Get the device pointer as an integer"
    return device_ctypes_pointer(obj).value

def device_ctypes_pointer(obj):
    "Get the ctypes object for the device pointer"
    require_device_memory(obj)
    return obj.device_ctypes_pointer

def is_device_memory(obj):
    """All CUDA memory object is recognized as an instance with the attribute
    "__cuda_memory__" defined and its value evaluated to True.

    All CUDA memory object should also define an attribute named
    "device_pointer" which value is an int(or long) object carrying the pointer
    value of the device memory address.  This is not tested in this method.
    """
    return getattr(obj, '__cuda_memory__', False)

def require_device_memory(obj):
    """A sentry for methods that accept CUDA memory object.
    """
    if not is_device_memory(obj):
        raise Exception("Not a CUDA memory object.")

def device_memory_depends(devmem, *objs):
    """Add dependencies to the device memory.

    Mainly used for creating structures that points to other device memory,
    so that the referees are not GC and released.
    """
    depset = getattr(devmem, "_depends_", [])
    depset.extend(objs)
    devmem = depset

def host_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    driver = Driver()
    varargs = []

    if stream:
        fn = driver.cuMemcpyHtoDAsync
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemcpyHtoD

    error = fn(device_pointer(dst), host_pointer(src), size, *varargs)
    driver.check_error(error, "Failed to copy memory H->D")

def device_to_host(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    driver = Driver()
    varargs = []

    if stream:
        fn = driver.cuMemcpyDtoHAsync
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemcpyDtoH

    error = fn(host_pointer(dst), device_pointer(src), size, *varargs)
    driver.check_error(error, "Failed to copy memory D->H")


def device_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    driver = Driver()
    varargs = []

    if stream:
        fn = driver.cuMemcpyDtoDAsync
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemcpyDtoD

    error = fn(device_pointer(dst), device_pointer(src), size, *varargs)
    driver.check_error(error, "Failed to copy memory D->H")

def device_memset(dst, val, size, stream=0):
    """Memset on the device.
    If stream is not zero, asynchronous mode is used.

    dst: device memory
    val: byte value to be written
    size: number of byte to be written
    stream: a CUDA stream
    """
    driver = Driver()
    varargs = []

    if stream:
        fn = driver.cuMemsetD8Async
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemsetD8

    error = fn(device_pointer(dst), val, size, *varargs)
    driver.check_error(error, "Failed to memset")
