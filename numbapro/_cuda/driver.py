import sys, os
from ctypes import *

# CUDA specific typedefs
cu_device = c_int
cu_device_attribute = c_int # enum
cu_context = c_void_p # an opaque handle
cu_module = c_void_p # an opaque handle
cu_jit_option = c_int # enum
cu_function = c_void_p  # an opaque handle
cu_device_ptr = c_void_p # defined as unsigned int on 32-bit and unsigned long long on 64-bit machine
cu_stream = c_int

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
        'cuDeviceGetCount':     (c_int, c_int),

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
        'cuMemcpyHtoD':         (c_int, cu_device_ptr, c_size_t),

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

        if not overide_path: # Try to discover cuda driver automatically
            # Determine platform and path of cuda driver
            if sys.platform == 'win32':
                dlloader = WinDLL
                path = '\\windows\\system32\\nvcuda.dll'
            else:
                dlloader = CDLL
                path = '/usr/lib/libcuda.so'

            # Environment variable always overide if present
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
        self.cuInit(0)

    def _cu_symbol_newer(self, symbol):
        try:
            return getattr(self.driver, '%s_v2' % symbol)
        except AttributeError:
            return getattr(self.driver, symbol)

