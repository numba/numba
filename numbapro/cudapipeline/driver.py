'''
This is more-or-less a object-oriented interface to the CUDA driver API.
It properly has a lot of resemblence with PyCUDA.
'''

import sys, os, warnings
import contextlib
from ctypes import *
from .error import *
from numbapro._utils import finalizer, mviewbuf
import threading
from weakref import WeakValueDictionary

#------------------
# Configuration

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
cu_function = c_void_p          # an opaque handle
cu_device_ptr = c_size_t        # defined as unsigned int on 32-bit
                                # and unsigned long long on 64-bit machine
cu_stream = c_void_p            # an opaque handle
cu_event = c_void_p

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


def _build_reverse_error_map():
    import sys
    prefix = 'CUDA_ERROR'
    module = sys.modules[__name__]
    return dict((getattr(module, i), i)
                for i in filter(lambda x: x.startswith(prefix), globals()))

# Expose bytearray creation
bytearray_from = pythonapi.PyByteArray_FromStringAndSize
bytearray_from.restype = py_object
bytearray_from.argtypes = c_void_p, c_ssize_t

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

        # CUresult cuDeviceGetCount(int *count);
        'cuDeviceGetCount':     (c_int, POINTER(c_int)),

        # CUresult cuDeviceGet(CUdevice *device, int ordinal);
        'cuDeviceGet':          (c_int, POINTER(cu_device), c_int),

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

        # CUresult cuCtxGetCurrent	(CUcontext *pctx);
        'cuCtxGetCurrent':      (c_int, POINTER(cu_context)),

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
    }

    OLD_API_PROTOTYPES = {
        # CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
        'cuFuncSetBlockShape': (c_int, cu_function, c_int, c_int, c_int),

        # CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
        'cuFuncSetSharedSize': (c_int, cu_function, c_uint),

        # CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
        'cuLaunchGrid':        (c_int, cu_function, c_int, c_int),

        # CUresult cuLaunchGridAsync(CUfunction f, int grid_width,
        #                            int grid_height, CUstream hStream);
        'cuLaunchGridAsync':   (c_int, cu_function, c_int, c_int, cu_stream),

        # CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
        'cuParamSetSize':      (c_int, cu_function, c_uint),

        # CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr,
        #                      unsigned int numbytes);
        'cuParamSetv':         (c_int, cu_function, c_int, c_void_p, c_uint),
    }

    NOT_IN_OLD_API = ['cuLaunchKernel']

    _REVERSE_ERROR_MAP = _build_reverse_error_map()

    __INSTANCE = None

    # A mapping from context handle -> Context instance
    _CONTEXTS = {}

    # Thread local storage for cache the context
    _THREAD_LOCAL = threading.local()

    def __new__(cls, override_path=None):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = inst = object.__new__(Driver)
            inst.old_api = False

            # Determine DLL type
            if sys.platform == 'win32':
                dlloader = WinDLL
                dldir = ['\\windows\\system32']
                dlname = 'nvcuda.dll'
            elif sys.platform == 'darwin':
                dlloader = CDLL
                dldir = ['/usr/local/cuda/lib']
                dlname = 'libcuda.dylib'
            else:
                dlloader = CDLL
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
            for path in candidates:
                try:
                    inst.driver = dlloader(path)
                    inst.path = path
                except OSError:
                    pass # can't find it, continue
                else:
                    break # got it; break out
            else:
                # not found, barf
                cls._raise_driver_not_found()

            # Obtain function pointers
            for func, prototype in inst.API_PROTOTYPES.items():
                restype = prototype[0]
                argtypes = prototype[1:]
                try:
                    ct_func = inst._cu_symbol_newer(func)
                except AttributeError:
                    if func in inst.NOT_IN_OLD_API:
                        # Symbol not found and is not in the old API?
                        # This indicates the driver is old
                        inst.old_api = True
                else:
                    ct_func.restype = restype
                    ct_func.argtypes = argtypes
                    setattr(inst, func, ct_func)

            if inst.old_api:
                # Old API, primiarily in Ocelot
                for func, prototype in inst.OLD_API_PROTOTYPES.items():
                    restype = prototype[0]
                    argtypes = prototype[1:]
                    ct_func = inst._cu_symbol_newer(func)
                    ct_func.restype = restype
                    ct_func.argtypes = argtypes
                    setattr(inst, func, ct_func)

            # initialize the API
            try:
                error = inst.cuInit(0)
                inst.check_error(error, "Failed to initialize CUDA driver")
            except AttributeError:
                # Not a real driver?
                cls._raise_driver_not_found()
            except CudaDriverError:
                # it doesn't work?
                cls._raise_driver_not_found()

        return cls.__INSTANCE

    @classmethod
    def _raise_driver_not_found(cls):
        cls.__INSTANCE = None # posion
        raise CudaSupportError(
                   "CUDA is not supported or the library cannot be found. "
                   "Try setting environment variable NUMBAPRO_CUDA_DRIVER "
                   "with the path of the CUDA driver shared library.")

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

    def check_error(self, error, msg, exit=False):
        if error:
            exc = CudaDriverError(msg, self._REVERSE_ERROR_MAP[error])
            if exit:
                print>>sys.stderr, exc
                sys.exit(1)
            else:
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

    def current_context(self, noraise=False):
        '''Get current context from TLS
        '''
        try:
            handle = self._THREAD_LOCAL.context
        except AttributeError:
            ctxt = self._get_current_context(noraise=noraise)
            if ctxt:
                self._cache_current_context(ctxt)
            return ctxt
        else:
            return self._CONTEXTS[handle]

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
        else:
            context = self._CONTEXTS[handle.value]
            return context

    def get_or_create_context(self, device=None):
        '''Returns the current context if exists, or get create a new one.
        '''
        try:
            return self.current_context()
        except CudaDriverError:
            return self.create_context(device)

    def release_context(self, context):
        handle = context._handle.value
        del self._CONTEXTS[handle]
        if handle == self._THREAD_LOCAL.context:
            # Remove all thread local
            for k in vars(self._THREAD_LOCAL).keys():
                delattr(self._THREAD_LOCAL, k)

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
    }

    def __init__(self, device_id):
        self.driver = Driver()
        got_device = c_int()
        error = self.driver.cuDeviceGet(byref(got_device), device_id)
        self.driver.check_error(error, 'Failed to get device %d' % device_id)
        assert device_id == got_device.value
        self.id = got_device.value
        self.__read_attributes()

    def __str__(self):
        return "CUDA device %d" % self.id

    def get_memory_info(self):
        "Returns (free, total) memory in bytes on the device."
        free = c_size_t()
        total = c_size_t()
        error = self.driver.cuMemGetInfo(byref(free), byref(total))
        msg = 'Failed to get memory info on device %d' % self.id
        self.driver.check_error(error, msg)
        return free.value, total.value

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

class _Context(finalizer.OwnerMixin):
    def __init__(self, device):
        self.device = device
        self._handle = cu_context()
        flags = 0
        if self.device.CAN_MAP_HOST_MEMORY:
            flags |= CU_CTX_MAP_HOST
        error = self.driver.cuCtxCreate(byref(self._handle), flags,
                                        self.device.id)
        self.driver.check_error(error,
                                'Failed to create context on %s' % self.device)
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error = driver.cuCtxDestroy(handle)
        driver.check_error(error, 'Failed to destroy context %s' % handle,
                           exit=True)

    @property
    def driver(self):
        return self.device.driver

    def __str__(self):
        return 'Context %s on %s' % (id(self), self.device)

class Stream(finalizer.OwnerMixin):
    def __init__(self):
        self.device = self.driver.current_context().device
        self._handle = cu_stream()
        error = self.driver.cuStreamCreate(byref(self._handle), 0)
        msg = 'Failed to create stream on %s' % self.device
        self.driver.check_error(error, msg)
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error = driver.cuStreamDestroy(handle)
        driver.check_error(error, 'Failed to destory stream %s' % handle,
                           exit=True)

    def __str__(self):
        return 'Stream %d on %s' % (self, self.device)

    def __int__(self):
        return self._handle.value

    def synchronize(self):
        self.driver.cuStreamSynchronize(self._handle)

    @contextlib.contextmanager
    def auto_synchronize(self):
        '''Use this for create a context that synchronize automatically.
        '''
        yield self
        self.synchronize()

    @property
    def driver(self):
        return self.device.driver


class HostAllocMemory(finalizer.OwnerMixin):
    __cuda_memory__ = True
    def __init__(self, bytesize, map=False, portable=False, wc=False):
        self.device = self.driver.current_context().device
        self.bytesize = bytesize
        self._handle = c_void_p()
        flags = 0
        if map:
            flags |= CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= CU_MEMHOSTALLOC_WRITECOMBINED
        error = self.driver.cuMemHostAlloc(byref(self._handle),
                                      bytesize, flags)
        self.driver.check_error(error, 'Failed to host alloc')
        self._finalizer_track(self._handle)

        self.devmem = cu_device_ptr()

        error = self.driver.cuMemHostGetDevicePointer(byref(self.devmem),
                                                      self._handle, 0)
        self.driver.check_error(error, 'Failed to get device ptr from host ptr')

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        driver.cuMemFreeHost(handle)

    def get_host_buffer(self):
        return bytearray_from(self._handle, self.bytesize)

    @property
    def device_pointer(self):
        return self._handle.value

    @property
    def driver(self):
        return Driver()


class DeviceMemory(finalizer.OwnerMixin):
    __cuda_memory__ = True
    def __init__(self, bytesize):
        self.device = self.driver.current_context().device
        self._allocate(bytesize)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error = driver.cuMemFree(handle)
        driver.check_error(error, 'Failed to free memory', exit=True)
        if debug_memory:
            global debug_memory_free
            debug_memory_free += 1

    def _allocate(self, bytesize):
        assert not hasattr(self, '_handle')
        self._handle = cu_device_ptr()
        error = self.driver.cuMemAlloc(byref(self._handle), bytesize)
        self.driver.check_error(error, 'Failed to allocate memory')
        self._finalizer_track(self._handle)
        self.bytesize = bytesize

        if debug_memory:
            global debug_memory_alloc
            debug_memory_alloc += 1

    @property
    def device_pointer(self):
        return self._handle.value

    @property
    def driver(self):
        return Driver()

class DeviceView(object):
    __cuda_memory__ = True
    def __init__(self, owner, offset):
        self.device = self.driver.current_context().device
        self._handle = cu_device_ptr(owner.device_pointer + offset)
        self._owner = owner

    @property
    def device_pointer(self):
        return self._handle.value
    
    @property
    def driver(self):
        return Driver()

class PinnedMemory(finalizer.OwnerMixin):
    __cuda_memory__ = True
    def __init__(self, owner, ptr, size, mapped):
        self._owner = owner
        self.device = self.driver.current_context().device
        if mapped and not self.device.CAN_MAP_HOST_MEMORY:
            raise CudaDriverError("Device %s cannot map host memory" %
                                  self.device)

        self._pointer = ptr
        # possible flags are portable (between context)
        # and deivce-map (map host memory to device thus no need
        # for memory transfer).
        flags = 0
        self._mapped = mapped
        if mapped:
            flags |= CU_MEMHOSTREGISTER_DEVICEMAP
        error = self.driver.cuMemHostRegister(ptr, size, flags)
        self.driver.check_error(error, 'Failed to pin memory')
        self._finalizer_track(self._pointer)
        self._get_device_pointer()

    def _get_device_pointer(self):
        assert self._mapped
        self._devmem = cu_device_ptr(0)
        flags = 0 # must be zero for now
        error = self.driver.cuMemHostGetDevicePointer(byref(self._devmem),
                                                      self._pointer, flags)
        self.driver.check_error(error, 'Failed to get device ptr from host ptr')

    @property
    def device_pointer(self):
        return self._devmem.value
    
    @classmethod
    def _finalize(cls, pointer):
        driver = Driver()
        error = driver.cuMemHostUnregister(pointer)
        driver.check_error(error, 'Failed to unpin memory')

    @property
    def driver(self):
        return Driver()

CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4

class Module(finalizer.OwnerMixin):
    def __init__(self, ptx):
        self.device = self.driver.current_context().device
        self.ptx = ptx

        self._handle = cu_module()
        ptx = c_char_p(self.ptx)

        info_log_n = 256
        c_info_log_n = c_int(info_log_n)
        c_info_log_buffer = (c_char * info_log_n)()

        option_keys = [CU_JIT_INFO_LOG_BUFFER,
                       CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES]
        option_vals = [cast(c_info_log_buffer, c_void_p),
                       addressof(c_info_log_n)]
        option_n = len(option_keys)
        c_option_keys = (c_int * option_n)(*option_keys)
        c_option_vals = (c_void_p * option_n)(*option_vals)

        error = self.driver.cuModuleLoadDataEx(byref(self._handle), ptx,
                                               option_n, c_option_keys,
                                               c_option_vals)

        self.driver.check_error(error, 'Failed to load module')
        self._finalizer_track(self._handle)

        self.info_log = c_info_log_buffer[:c_info_log_n.value]

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error =  driver.cuModuleUnload(handle)
        driver.check_error(error, 'Failed to unload module', exit=True)

    @property
    def driver(self):
        return self.device.driver

class Function(finalizer.OwnerMixin):

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
        return self.device.driver

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
        assert self.driver.current_context().device is self.device
        launch_kernel(self._handle, self.griddim, self.blockdim,
                      self.sharedmem, self.stream, args)

class Event(finalizer.OwnerMixin):
    def __init__(self, timing=True):
        self.device = self.driver.current_context().device
        self._handle = cu_event()
        self.timing = timing
        flags = 0
        if not timing:
            flags |= CU_EVENT_DISABLE_TIMING
        error = self.driver.cuEventCreate(byref(self._handle), flags)
        self.driver.check_error(error, 'Failed to create event')
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error =  driver.cuEventDestroy(handle)
        driver.check_error(error, 'Failed to unload module', exit=True)

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
        return self.device.driver


def event_elapsed_time(evtstart, evtend):
    driver = evtstart.driver
    msec = c_float()
    err = driver.cuEventElapsedTime(byref(msec), evtstart._handle,
                                    evtend._handle)
    driver.check_error(err, 'Failed to get elapsed time')
    return msec.value

def launch_kernel(cufunc_handle, griddim, blockdim, sharedmem, hstream, args):
    driver = Driver()
    if driver.old_api:
        error = driver.cuFuncSetBlockShape(cufunc_handle,  *blockdim)
        driver.check_error(error, "Failed to set block shape.")

        error = driver.cuFuncSetSharedSize(cufunc_handle, sharedmem)
        driver.check_error(error, "Failed to set shared memory size.")

        # count parameter byte size
        bytesize = 0
        for arg in args:
            if isinstance(arg, AllocatedDeviceMemory):
                size = sizeof(arg._handle)
            else:
                size = sizeof(arg)
            bytesize += size

        error = driver.cuParamSetSize(cufunc_handle, bytesize)
        driver.check_error(error, 'Failed to set parameter size (%d)' % bytesize)

        offset = 0
        for i, arg in enumerate(args):
            if isinstance(arg, AllocatedDeviceMemory):
                size = sizeof(arg._handle)
                error = driver.cuParamSetv(cufunc_handle, offset,
                                           addressof(arg._handle),
                                           size)
            else:
                size = sizeof(arg)
                error = driver.cuParamSetv(cufunc_handle, offset,
                                           addressof(arg), size)
            driver.check_error(error, 'Failed to set parameter %d' % i)
            offset += size


        gx, gy, _ = griddim
        error = driver.cuLaunchGridAsync(cufunc_handle, gx, gy, hstream)

        driver.check_error(error, 'Failed to launch kernel')
    else:
        gx, gy, gz = griddim
        bx, by, bz = blockdim

        param_vals = []
        for arg in args:
            if isinstance(arg, AllocatedDeviceMemory):
                param_vals.append(addressof(arg._handle))
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

def require_context(fn):
    "Decorator"
    def _wrapper(*args, **kws):
        get_or_create_context()
        return fn(*args, **kws)
    return _wrapper

#
# CUDA memory functions
#

def host_pointer(obj):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous 
    completes.
    """
    mv = memoryview(obj)
    return mviewbuf.memoryview_get_buffer(mv)

def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    mv = memoryview(obj)
    return mviewbuf.memoryview_get_extents(mv)

def host_memory_size(obj):
    "Get the size of the memory"
    s, e = host_memory_extents(obj)
    assert e >= s
    return e - s

def device_pointer(obj):
    require_device_memory(obj)
    return obj.device_pointer

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

def host_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous 
    completes.
    """
    driver = dst.driver
    varargs = []

    if stream:
        fn = driver.cuMemcpyHtoDAsync
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemcpyHtoD

    devptr = dst.device_pointer
    hostptr = host_pointer(src)
    
    error = fn(device_pointer(dst), host_pointer(src), size, *varargs)
    driver.check_error(error, "Failed to copy memory H->D")

def device_to_host(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous 
    completes.
    """
    driver = src.driver
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
    driver = src.driver
    varargs = []

    if stream:
        fn = driver.cuMemcpyDtoDAsync
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemcpyDtoD

    error = fn(device_pointer(dst), device_pointer(src), size, *varargs)
    driver.check_error(error, "Failed to copy memory D->H")

def device_memset(dst, val, size, stream=0):
    driver = dst.driver
    varargs = []

    if stream:
        fn = driver.cuMemsetD8Async
        varargs.append(stream._handle)
    else:
        fn = driver.cuMemsetD8

    error = fn(device_pointer(dst), val, size, *varargs)
    driver.check_error(error, "Failed to memset")

