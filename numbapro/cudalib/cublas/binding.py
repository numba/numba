import sys
import numpy as np
from ctypes import *

from numbapro.cudalib.libutils import Lib, ctype_function
from numbapro.cudapipeline.driver import cu_stream
from numbapro._utils import finalizer
from numbapro import cuda

INV_STATUS = dict(
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14
)

STATUS = dict((v, k) for k, v in INV_STATUS.items())

cublasStatus_t = c_int

CUBLAS_FILL_MODE_LOWER=0
CUBLAS_FILL_MODE_UPPER=1

cublasFillMode_t = c_int

CUBLAS_DIAG_NON_UNIT=0
CUBLAS_DIAG_UNIT=1

cublasDiagType_t = c_int

CUBLAS_SIDE_LEFT =0
CUBLAS_SIDE_RIGHT=1

cublasSideMode_t = c_int

CUBLAS_OP_N=0
CUBLAS_OP_T=1
CUBLAS_OP_C=2

cublasOperation_t = c_int

CUBLAS_POINTER_MODE_HOST   = 0
CUBLAS_POINTER_MODE_DEVICE = 1

cublasPointerMode_t = c_int

CUBLAS_ATOMICS_NOT_ALLOWED   = 0
CUBLAS_ATOMICS_ALLOWED       = 1        

cublasAtomicsMode_t = c_int

cublasHandle_t = c_void_p # opaque handle

class CuBLASError(Exception):
    def __init__(self, code):
        super(CuBLASError, self).__init__(STATUS[code])

class libcublas(Lib):
    lib = 'libcublas'
    ErrorType = CuBLASError

    cublasCreate_v2 = ctype_function(cublasStatus_t,
                                     POINTER(cublasHandle_t))  # handle

    cublasDestroy_v2 = ctype_function(cublasStatus_t,
                                      cublasHandle_t)   # handle

    cublasGetVersion_v2 = ctype_function(cublasStatus_t,
                                         cublasHandle_t, # handle
                                         POINTER(c_int)) # version

    cublasSetStream_v2 = ctype_function(cublasStatus_t,
                                        cublasHandle_t, # handle
                                        cu_stream)      # streamId

    cublasGetStream_v2 = ctype_function(cublasStatus_t,
                                        cublasHandle_t, # handle
                                        POINTER(cu_stream))      # streamId

    cublasGetPointerMode_v2 = ctype_function(cublasStatus_t,
                                             cublasHandle_t,            # handle
                                             POINTER(cublasPointerMode_t)) # mode

    cublasSetPointerMode_v2 = ctype_function(cublasStatus_t,
                                             cublasHandle_t,      # handle
                                             cublasPointerMode_t) # mode

    cublasGetAtomicsMode = ctype_function(cublasStatus_t,
                                          cublasHandle_t, # handle
                                          POINTER(cublasAtomicsMode_t)) # mode

    cublasSetAtomicsMode = ctype_function(cublasStatus_t,
                                          cublasHandle_t,       # handle
                                          cublasAtomicsMode_t)  # mode

    # Level 1
    cublasSnrm2_v2 = ctype_function(cublasStatus_t,
                            cublasHandle_t, # handle
                            c_int,          # n
                            c_void_p,       # device array
                            c_int,          # incx
                            c_void_p)       # result - host/device scalar

    cublasDnrm2_v2 = cublasSnrm2_v2
    cublasScnrm2_v2 = cublasSnrm2_v2
    cublasDznrm2_v2 = cublasSnrm2_v2


class c_complex(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]

class c_double_complex(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]

def _Tnrm2(fmt, cty):
    def _wrapped(self, n, x, incx):
        result = cty()
        fn = getattr(self._api, 'cublas%snrm2_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           byref(result))
        if cty in (c_complex, c_double_complex):
            return complex(result.real.value, result.imag.value)
        else:
            return result.value
    return _wrapped

class cuBlas(finalizer.OwnerMixin):
    def __init__(self):
        self._api = libcublas()
        self._handle = cublasHandle_t()
        self._api.cublasCreate_v2(byref(self._handle))
        self._finalizer_track((self._handle, self._api))
        self._stream = 0

    @classmethod
    def _finalize(self, res):
        handle, api = res
        api.cublasDestroy_v2(handle)
        
    @property
    def version(self):
        ver = c_int()
        self._api.cublasGetVersion_v2(self._handle, byref(ver))
        return ver.value

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        self._stream = stream
        self._api.cublasSetStream_v2(self._handle, self._stream._handle)

    @property
    def pointer_mode(self):
        mode = cublasPointerMode_t()
        self._api.cublasGetPointerMode_v2(self._handle, byref(mode))
        return mode.value

    @pointer_mode.setter
    def pointer_mode(self, mode):
        self._api.cublasSetPointerMode_v2(self._handle, int(mode))

    @property
    def atomics_mode(self):
        mode = cublasAtomicsMode_t()
        self._api.cublasGetAtomicsMode(self._handle, byref(mode))
        return mode.value

    @atomics_mode.setter
    def atomics_mode(self, mode):
        self._api.cublasSetAtomicsMode(self._handle, int(mode))

    # Level 1

    Snrm2 = _Tnrm2('S', c_float)
    Dnrm2 = _Tnrm2('D', c_double)
    Scnrm2 = _Tnrm2('Sc', c_float)
    Dznrm2 = _Tnrm2('Dz', c_double)


def _get_data_ptr(ary):
    if isinstance(ary, cuda.DeviceArrayBase):
        resptr = ary.device_raw_ptr.value
    else:
        resptr = ary.ctypes.data
    return resptr
