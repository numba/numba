import sys, os
from ctypes import *

# CUDA specific typedefs
cu_device = c_int
cu_device_attribute = c_int # enum
cu_context = c_void_p # an opaque handle
cu_module = c_void_p # an opaque handle
cu_jit_option = c_int # enum
cu_function = c_void_p  # an opaque handle
cu_device_ptr = c_size_t # defined as unsigned int on 32-bit and unsigned long long on 64-bit machine
cu_stream = c_int

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



def _build_reverse_error_map():
    import sys
    prefix = 'CUDA_ERROR'
    module = sys.modules[__name__]
    return dict((getattr(module, i), i)
                for i in filter(lambda x: x.startswith(prefix), globals()))

_REVERSE_ERROR_MAP = _build_reverse_error_map()

class DriverError(Exception):
    pass

def _check_error(error, msg):
    if error != CUDA_SUCCESS:
        raise DriverError(msg, _REVERSE_ERROR_MAP[error])

class Driver(object):
    '''Facade to the CUDA Driver API.
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

        # CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
        'cuMemAlloc':         (c_int, POINTER(cu_device_ptr), c_size_t),

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
        'cuStreamCreate':       (c_int, cu_stream, c_uint),

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

    def __init__(self, overide_path=None):
        self.old_api = False

        # Determine DLL type
        if sys.platform == 'win32':
            dlloader = WinDLL
            path = '\\windows\\system32\\nvcuda.dll'
        else:
            dlloader = CDLL
            path = '/usr/lib/libcuda.so'

        if not overide_path: # Try to discover cuda driver automatically
            # Environment variable always overide if present
            # and overide_path is not defined.
            path = os.environ.get('NUMBAPRO_CUDA_DRIVER', path)
        else:
            path = overide_path

        # Load the driver
        try:
            self.driver = dlloader(path)
        except OSError:
            raise ImportError(
                      "CUDA is not supported or the library cannot be found. "
                      "Try setting environment variable NUMBAPRO_CUDA_DRIVER "
                      "with the path of the CUDA driver shared library.")

        # Obtain function pointers
        for func, prototype in self.API_PROTOTYPES.items():
            restype = prototype[0]
            argtypes = prototype[1:]
            try:
                ct_func = self._cu_symbol_newer(func)
            except AttributeError:
                if func in self.NOT_IN_OLD_API:
                    self.old_api = True
            else:
                ct_func.restype = restype
                ct_func.argtypes = argtypes
                setattr(self, func, ct_func)

        if self.old_api:
            # Old API, primiarily in Ocelot
            for func, prototype in self.OLD_API_PROTOTYPES.items():
                restype = prototype[0]
                argtypes = prototype[1:]
                ct_func = self._cu_symbol_newer(func)
                ct_func.restype = restype
                ct_func.argtypes = argtypes
                setattr(self, func, ct_func)

        # initialize the API
        error = self.cuInit(0)
        _check_error(error, "Failed to initialize CUDA driver")

    def _cu_symbol_newer(self, symbol):
        try:
            return getattr(self.driver, '%s_v2' % symbol)
        except AttributeError:
            return getattr(self.driver, symbol)

    def get_device_count(self):
        count = c_int()
        error = self.cuDeviceGetCount(byref(count))
        _check_error(error, 'Failed to get number of device')
        return count.value

CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

class Device(object):

    ATTRIBUTES = {
      'MAX_THREADS_PER_BLOCK': CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      'MAX_GRID_DIM_X':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      'MAX_GRID_DIM_Y':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      'MAX_GRID_DIM_Z':        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      'MAX_BLOCK_DIM_X':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
      'MAX_BLOCK_DIM_Y':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
      'MAX_BLOCK_DIM_Z':       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      'MAX_SHARED_MEMORY':     CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    }

    def __init__(self, driver, device_id):
        got_device = c_int()
        error = driver.cuDeviceGet(byref(got_device), device_id)
        _check_error(error, 'Failed to get device %d')
        assert device_id == got_device.value
        self.driver = driver
        self.id = got_device.value
        self._read_attributes()

    def __str__(self):
        return "CUDA device %d" % self.id

    def _read_attributes(self):
        got_value = c_int()
        for name, num in self.ATTRIBUTES.items():
            error = self.driver.cuDeviceGetAttribute(byref(got_value), num,
                                                     self.id)
            _check_error(error, 'Failed to read attribute "%s" from %s' % (name, self))
            setattr(self, name, got_value.value)

        got_major = c_int()
        got_minor = c_int()
        error = self.driver.cuDeviceComputeCapability(byref(got_major),
                                                      byref(got_minor),
                                                      self.id);
        _check_error(error, 'Failed to read compute capability from %s' % self)

        setattr(self, 'COMPUTE_CAPABILITY', (got_major.value, got_minor.value))

    @property
    def attributes(self):
        '''Returns all attributes as a dictionary
        '''
        keys = list(self.ATTRIBUTES.keys())
        keys += ['COMPUTE_CAPABILITY']
        return dict((k, getattr(self, k)) for k in keys)


class Context(object):
    def __init__(self, device):
        self.device = device
        self._handle = cu_context()
        error = self.driver.cuCtxCreate(byref(self._handle), 0, self.device.id)
        _check_error(error, 'Failed to create context on %s' % self.device)

    def __del__(self):
        error = self.driver.cuCtxDestroy(self._handle)
        _check_error(error, 'Failed to destroy context on %s' % self.device)

    @property
    def driver(self):
        return self.device.driver

class DeviceMemory(object):
    def __init__(self, context, bytesize):
        self.context = context
        self._handle = cu_device_ptr()
        error = self.driver.cuMemAlloc(byref(self._handle), bytesize)
        _check_error(error, 'Failed to allocate memory')

    def __del__(self):
        error = self.driver.cuMemFree(self._handle)
        _check_error(error, 'Failed to free memory')

    def to_device_raw(self, src, size, stream=0):
        if stream:
            self.driver.cuMemcpyHtoD(self._handle, src, size, stream)
        else:
            self.driver.cuMemcpyHtoD(self._handle, src, size)

    def from_device_raw(self, dst, size, stream=0):
        if stream:
            self.driver.cuMemcpyDtoH(dst, self._handle, size, stream)
        else:
            self.driver.cuMemcpyDtoH(dst, self._handle, size)

    @property
    def driver(self):
        return self.device.driver

    @property
    def device(self):
        return self.context.device


CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4

class Module(object):
    def __init__(self, context, ptx):
        self.context = context
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
        _check_error(error, 'Failed to load module')

        self.info_log = c_info_log_buffer

    def __del__(self):
        error =  self.driver.cuModuleUnload(self._handle)
        _check_error(error, 'Failed to unload module')

    @property
    def driver(self):
        return self.device.driver

    @property
    def device(self):
        return self.context.device


class Function(object):

    griddim = 1, 1, 1
    blockdim = 1, 1, 1
    stream = 0
    sharedmem = 0

    def __init__(self, module, name):
        self.module = module
        self.name = name
        self._handle = cu_function()
        error = self.driver.cuModuleGetFunction(byref(self._handle),
                                                self.module._handle,
                                                name);
        _check_error(error, 'Failed to get function "%s" from module' % name)

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
        return 'CUDA kernel %s' % self.name

    def configure(self, griddim, blockdim, sharedmem=0, stream=0):
        import copy

        while len(griddim) < 3:
            griddim += (1,)

        while len(blockdim) < 3:
            blockdim += (1,)

        inst = copy.copy(self) # shallow clone the object
        inst.griddim = griddim
        inst.blockdim = blockdim
        inst.sharedmem = sharedmem
        inst.stream = stream
        return inst

    def __call__(self, *args):
        '''
        *args -- Must be either ctype objects of DevicePointer instances.
        '''
        if self.driver.old_api:
            error = self.driver.cuFuncSetBlockShape(self._handle,  *self.blockdim)
            _check_error(error, "Failed to set block shape.")

            error = self.driver.cuFuncSetSharedSize(self._handle, self.sharedmem)
            _check_error(error, "Failed to set shared memory size.")

            # count parameter byte size
            bytesize = 0
            for arg in args:
                if isinstance(arg, DeviceMemory):
                    size = sizeof(arg._handle)
                else:
                    size = sizeof(arg)
                bytesize += size

            error = self.driver.cuParamSetSize(self._handle, bytesize)
            _check_error(error, 'Failed to set parameter size (%d)' % bytesize)

            offset = 0
            for i, arg in enumerate(args):
                if isinstance(arg, DeviceMemory):
                    size = sizeof(arg._handle)
                    error = self.driver.cuParamSetv(self._handle, offset,
                                                    addressof(arg._handle),
                                                    size)
                else:
                    size = sizeof(arg)
                    error = self.driver.cuParamSetv(arg, offset, addressof(arg),
                                                    size)
                _check_error(error, 'Failed to set parameter %d' % i)
                offset += size


            gx, gy, _ = self.griddim
            if self.stream:
                error = self.driver.cuLaunchGrid(self._handle, gx, gy, stream)
            else:
                error = self.driver.cuLaunchGrid(self._handle, gx, gy)

            _check_error(error, 'Failed to launch kernel')

        else:

            gx, gy, gz = self.griddim
            bx, by, bz = self.blockdim

            param_vals = []
            for arg in args:
                if isinstance(arg, DeviceMemory):
                    param_vals.append(addressof(arg._handle))
                else:
                    param_vals.append(addressof(arg))
            params = (c_void_p * len(param_vals))(*param_vals)

            error = self.driver.cuLaunchKernel(
                        self._handle,
                        gx, gy, gz,
                        bx, by, bz,
                        self.sharedmem, self.stream,
                        # XXX: Why does the following line cannot be changed
                        #      to a variable of the same value in 64-bit Linux?
                        #      A python bug?
                        cast(addressof(params), POINTER(c_void_p)),
                        None)

            _check_error(error, "Failed to launch kernel")


