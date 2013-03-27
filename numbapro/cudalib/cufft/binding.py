import sys
import numpy as np
from ctypes import *

from numbapro.cudalib.libutils import Lib, ctype_function
from numbapro.cudapipeline.driver import cu_stream
from numbapro._utils import finalizer

STATUS = dict(
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9
)

cufftResult = c_int

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

CUFFT_R2C = 0x2a     # Real to Complex (interleaved)
CUFFT_C2R = 0x2c     # Complex (interleaved) to Real
CUFFT_C2C = 0x29     # Complex to Complex, interleaved
CUFFT_D2Z = 0x6a     # Double to Double-Complex
CUFFT_Z2D = 0x6c     # Double-Complex to Double
CUFFT_Z2Z = 0x69      # Double-Complex to Double-Complex

cufftType = c_int

CUFFT_COMPATIBILITY_NATIVE          = 0x00
CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01    # The default value
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
CUFFT_COMPATIBILITY_FFTW_ALL        = 0x03

CUFFT_COMPATIBILITY_DEFAULT         = CUFFT_COMPATIBILITY_FFTW_PADDING

cufftCompatibility = c_int

cufftHandle = c_int

class CuFFTError(Exception):
    def __init__(self, code):
        super(CuFFTError, self).__init__(STATUS[code])

class libcufft(Lib):
    lib = 'libcufft'
    ErrorType = CuFFTError

    @property
    def version(self):
        ver = c_int(0)
        self.cufftGetVersion(byref(ver))
        return ver.value

    cufftGetVersion = ctype_function(cufftResult, POINTER(c_int))

    cufftPlan1d = ctype_function(cufftResult,
                                 POINTER(cufftHandle), # plan
                                 c_int,                # nx
                                 cufftType,            # type
                                 c_int, # batch - deprecated - use cufftPlanMany
                                 )

    cufftPlan2d = ctype_function(cufftResult,
                                 POINTER(cufftHandle), # plan
                                 c_int,                # nx
                                 c_int,                # ny
                                 cufftType             # type
                                 )

    cufftPlan3d = ctype_function(cufftResult,
                                 POINTER(cufftHandle),  # plan
                                 c_int,                 # nx
                                 c_int,                 # ny
                                 c_int,                 # nz
                                 cufftType              # type
                                 )

    cufftPlanMany = ctype_function(cufftResult,
                                   POINTER(cufftHandle), # plan
                                   c_int,                # rank
                                   POINTER(c_int),       # n
                                   POINTER(c_int),       # inembed
                                   c_int,                # istride
                                   c_int,                # idist
                                   POINTER(c_int),       # onembed
                                   c_int,                # ostride
                                   c_int,                # odist
                                   cufftType,            # type
                                   c_int,                # batch
                                   )

    cufftDestroy = ctype_function(cufftResult,
                                  cufftHandle, # plan
                                  )

    cufftExecC2C = ctype_function(cufftResult,
                                  cufftHandle,              # plan
                                  c_void_p, # POINTER(cufftComplex) idata
                                  c_void_p, # POINTER(cufftComplex) odata
                                  c_int                     # direction
                                  )

    cufftExecR2C = ctype_function(cufftResult,
                                  cufftHandle,           # plan
                                  c_void_p, # POINTER(cufftReal) idata
                                  c_void_p, # POINTER(cufftComplex) odata
                                  )

    cufftExecC2R = ctype_function(cufftResult,
                                  cufftHandle,              # plan
                                  c_void_p, # POINTER(cufftComplex) idata
                                  c_void_p, # POINTER(cufftReal) odata
                                  )

    cufftExecZ2Z = ctype_function(cufftResult,
                                  cufftHandle,                 # plan
                                  c_void_p, # POINTER(cufftDoubleComplex) idata
                                  c_void_p, # POINTER(cufftDoubleComplex) odata
                                  c_int,                       # direction
                                  )

    cufftExecD2Z = ctype_function(cufftResult,
                                  cufftHandle,                 # plan
                                  c_void_p, # POINTER(cufftDoubleReal) idata
                                  c_void_p, # POINTER(cufftDoubleComplex) odata
                                  )

    cufftExecZ2D = ctype_function(cufftResult,
                                  cufftHandle,                 # plan
                                  c_void_p, # POINTER(cufftDoubleComplex) idata
                                  c_void_p, # POINTER(cufftDoubleReal) odata
                                  )

    cufftSetStream = ctype_function(cufftResult,
                                    cufftHandle,        # plan,
                                    cu_stream,          # stream
                                    )

    cufftSetCompatibilityMode = ctype_function(cufftResult,
                                               cufftHandle,         # plan,
                                               cufftCompatibility   # mode
                                               )
cufft_dtype_to_name = {
    CUFFT_R2C:'R2C',
    CUFFT_C2R:'C2R',
    CUFFT_C2C:'C2C',
    CUFFT_D2Z:'D2Z',
    CUFFT_Z2D:'Z2D',
    CUFFT_Z2Z:'Z2Z',
}

class Plan(finalizer.OwnerMixin):
    @classmethod
    def one(cls, dtype, nx):
        "cufftPlan1d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        BATCH = 1 # deprecated args to cufftPlan1d
        status = inst._api.cufftPlan1d(byref(inst._handle), int(nx), int(dtype),
                                       BATCH)
        inst.dtype = dtype
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def two(cls, dtype, nx, ny):
        "cufftPlan2d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        status = inst._api.cufftPlan2d(byref(inst._handle), int(nx), int(ny),
                                       int(dtype))
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def three(cls, dtype, nx, ny, nz):
        "cufftPlan3d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        status = inst._api.cufftPlan3d(byref(inst._handle), int(nx), int(ny),
                                       int(dtype))
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def many(cls, shape, dtype):
        "cufftPlanMany"
        raise NotImplementedError

    @classmethod
    def _finalize(cls, res):
        handle, api = res
        api.cufftDestroy(handle)

    def set_stream(self, stream):
        "Associate a CUDA stream to this plan object"
        return self._api.cufftSetStream(self._handle, stream)

    def exe(self, idata, odata, dir):
        meth = getattr(self._api, 'cufftExec' + cufft_dtype_to_name[self.dtype])
        return meth(self._handle, idata.device_raw_ptr.value,
                    odata.device_raw_ptr.value, int(dir))
