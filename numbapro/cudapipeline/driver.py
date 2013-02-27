'''
This is more-or-less a object-oriented interface to the CUDA driver API.
It properly has a lot of resemblence with PyCUDA.
'''

import sys, os, warnings
import contextlib
from ctypes import *
from .error import *
from numbapro._utils import finalizer
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

def _build_reverse_error_map():
    import sys
    prefix = 'CUDA_ERROR'
    module = sys.modules[__name__]
    return dict((getattr(module, i), i)
                for i in filter(lambda x: x.startswith(prefix), globals()))


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

        # CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
        'cuMemsetD8':         (c_int, cu_device_ptr, c_uint8, c_size_t),

        # CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
        #                          size_t N, CUstream hStream);
        'cuMemsetD8Async':    (c_int, cu_device_ptr, c_uint8, c_size_t, cu_stream),

        # CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
        #                       size_t ByteCount);
        'cuMemcpyHtoD':         (c_int, cu_device_ptr, c_void_p, c_size_t),

        # CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
        #                            size_t ByteCount, CUstream hStream);
        'cuMemcpyHtoDAsync':    (c_int, cu_device_ptr, c_void_p, c_size_t,
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
                dldir = '\\windows\\system32'
                dlname = 'nvcuda.dll'
            elif sys.platform == 'darwin':
                dlloader = CDLL
                dldir = '/usr/local/cuda/lib'
                dlname = 'libcuda.dylib'
            else:
                dlloader = CDLL
                dldir = '/usr/lib'
                dlname = 'libcuda.so'

            # First search for the name in the default library path.
            # If that is not found, try the specific path.
            candidates = [dlname, os.path.join(dldir, dlname)]

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
            if noraise:
                return None
            else:
                raise CudaDriverError("No context was created")
        else:
            return self._CONTEXTS[handle]

    def _cache_current_context(self, ctxt):
        '''Store current context into TLS
        '''
        self._THREAD_LOCAL.context = ctxt._handle.value

    def get_current_context(self):
        '''Uses CUDA driver API to get current context
        '''
        handle = cu_context()
        error = self.cuCtxGetCurrent(byref(handle))
        self.check_error(error, "Fail to get current context.")
        if handle.value is None:
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

    def __read_attributes(self):
        got_value = c_int()
        for name, num in self.ATTRIBUTES.items():
            error = self.driver.cuDeviceGetAttribute(byref(got_value), num,
                                                     self.id)
            self.driver.check_error(error, 'Failed to read attribute "%s" from %s' % (name, self))
            setattr(self, name, got_value.value)

        got_major = c_int()
        got_minor = c_int()
        error = self.driver.cuDeviceComputeCapability(byref(got_major),
                                                      byref(got_minor),
                                                      self.id)
        self.driver.check_error(error, 'Failed to read compute capability from %s' % self)

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
        if self.device.CAN_MAP_HOST_MEMORY:
            flags = CU_CTX_MAP_HOST
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
        self.context = Driver().current_context()
        self._handle = cu_stream()
        error = self.driver.cuStreamCreate(byref(self._handle), 0)
        self.driver.check_error(error, 'Failed to create stream on %s' % self.context)
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error = driver.cuStreamDestroy(handle)
        driver.check_error(error, 'Failed to destory stream %s' % handle,
                           exit=True)

    def __str__(self):
        return 'Stream %d on %s' % (self, self.context)

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

    @property
    def device(self):
        return self.context.device

class DevicePointer(object):
    '''Memory on the GPU deivce.

    The lifetime of the object is tied to the GPU memory handle; that is the
    memory is released when this object is released.
    '''
    def __init__(self, handle):
        self._handle = handle
        self._depends = []
        self.context = Driver().current_context()

    def to_device_raw(self, src, size, stream=None, offset=0):
        ptr = cu_device_ptr(self._handle.value + offset)
        if stream:
            error = self.driver.cuMemcpyHtoDAsync(ptr, src, size,
                                                  stream._handle)
        else:
            error = self.driver.cuMemcpyHtoD(ptr, src, size)
        self.driver.check_error(error, "Failed to copy memory H->D")

    def from_device_raw(self, dst, size, stream=None, offset=0):
        ptr = cu_device_ptr(self._handle.value + offset)
        if stream:
            error = self.driver.cuMemcpyDtoHAsync(dst, ptr, size,
                                                  stream._handle)
        else:
            error = self.driver.cuMemcpyDtoH(dst, ptr, size)
        self.driver.check_error(error, "Failed to copy memory D->H")

    def memset(self, val, size, stream=None):
        if stream:
            error = self.driver.cuMemsetD8Async(self._handle, val, size,
                                                stream._handle)
        else:
            error = self.driver.cuMemsetD8(self._handle, val, size)
        self.driver.check_error(error, "Failed to set memory")

    def add_dependencies(self, *args):
        self._depends.extend(args)

    @property
    def driver(self):
        return self.device.driver

    @property
    def device(self):
        return self.context.device

class AllocatedDeviceMemory(DevicePointer, finalizer.OwnerMixin):
    def __init__(self, bytesize=None):
        self._depends = []
        self.context = Driver().current_context()
        if bytesize is not None:
            self.allocate(bytesize)

    @classmethod
    def _finalize(cls, handle):
        driver = Driver()
        error = driver.cuMemFree(handle)
        driver.check_error(error, 'Failed to free memory', exit=True)
        if debug_memory:
            global debug_memory_free
            debug_memory_free += 1

    def allocate(self, bytesize):
        assert not hasattr(self, '_handle')
        self._handle = cu_device_ptr()
        error = self.driver.cuMemAlloc(byref(self._handle), bytesize)
        self.driver.check_error(error, 'Failed to allocate memory')
        self._finalizer_track(self._handle)
        
        if debug_memory:
            global debug_memory_alloc
            debug_memory_alloc += 1

    def offset(self, offset):
        handle = cu_device_ptr(self._handle.value + offset)
        # create new instance without allocation
        devmem = type(self)()
        devmem._handle = handle
        devmem.add_dependencies(self) # avoid free the parent pointer
        return devmem


class PinnedMemory(finalizer.OwnerMixin):

    # Use a weak value dictionary to cache pointer-value -> PinnedMemory object.
    __cache = WeakValueDictionary()

    def __new__(cls, ptr, size, mapped=False):
        if isinstance(ptr, int) or isinstance(ptr, long):
            ptr_value = ptr
        else:
            ptr_value = ptr.value
        if ptr_value in cls.__cache:
            # If the memory is already pinned,
            # return the existing object
            return cls.__cache[ptr_value]

        inst = object.__new__(PinnedMemory)
        # Cache instance in the cache
        cls.__cache[ptr_value] = inst
        inst.__initialize(ptr, size, mapped=mapped)
        return inst

    def __initialize(self, ptr, size, mapped):
        if mapped and not self.context.device.CAN_MAP_HOST_MEMORY:
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

    def get_device_pointer(self):
        assert self._mapped
        dptr = cu_device_ptr(0)
        hptr=  self._pointer
        flags = 0 # must be zero for now
        error = self.driver.cuMemHostGetDevicePointer(byref(dptr), hptr, flags)
        return DevicePointer(dptr)

    @classmethod
    def _finalize(cls, pointer):
        driver = Driver()
        error = driver.cuMemHostUnregister(pointer)
        driver.check_error(error, 'Failed to unpin memory')

    @property
    def context(self):
        return self.driver.current_context()

    @property
    def driver(self):
        return Driver()

CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4

class Module(finalizer.OwnerMixin):
    def __init__(self, ptx):
        self.context = Driver().current_context()
        self.ptx = ptx

        self._handle = cu_module()
        ptx = c_char_p(self.ptx)

        info_log_n = 256
        c_info_log_n = c_int(info_log_n)
        c_info_log_buffer = (c_char * info_log_n)()

        option_keys = [CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES]
        option_vals = [cast(c_info_log_buffer, c_void_p), addressof(c_info_log_n)]
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

    @property
    def device(self):
        return self.context.device


class Function(finalizer.OwnerMixin):

    griddim = 1, 1, 1
    blockdim = 1, 1, 1
    stream = None
    sharedmem = 0

    def __init__(self, module, name):
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
    def device(self):
        return self.context.device

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
        assert self.driver.current_context() is self.context
        launch_kernel(self._handle, self.griddim, self.blockdim,
                      self.sharedmem, self.stream, args)

def launch_kernel(cufunc_handle, griddim, blockdim, sharedmem, stream_handle, args):
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
                error = driver.cuParamSetv(cufunc_handle, offset, addressof(arg),
                                                size)
            driver.check_error(error, 'Failed to set parameter %d' % i)
            offset += size


        gx, gy, _ = griddim
        error = driver.cuLaunchGridAsync(cufunc_handle, gx, gy, stream_handle)

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
                    stream_handle,
                    params,
                    None)

        driver.check_error(error, "Failed to launch kernel")


def get_or_create_context():
    "Get the current context if it exists or create one."
    from .driver import Driver, Device
    driver = Driver()
    device_number = 0  # default device id
    cxt = driver.current_context(noraise=True)
    if cxt is None:
        device = Device(device_number)
        cxt = driver.create_context(device)
    return cxt

def require_context(fn):
    "Decorator"
    def _wrapper(*args, **kws):
        get_or_create_context()
        return fn(*args, **kws)
    return _wrapper

