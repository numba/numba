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

    cublasSdot_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t,     # handle
                                   c_int,              # n
                                   c_void_p,   # x
                                   c_int,              # incx
                                   c_void_p,   # y
                                   c_int,              # incy,
                                   c_void_p)   # result h/d ptr
    cublasDdot_v2 = cublasSdot_v2
    cublasCdotu_v2 = cublasSdot_v2
    cublasZdotu_v2 = cublasSdot_v2
    cublasCdotc_v2 = cublasSdot_v2
    cublasZdotc_v2 = cublasSdot_v2

    cublasSscal_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t,   # handle
                                    c_int,            # n
                                    c_void_p,         # alpha h/d
                                    c_void_p,         # x
                                    c_int)            # incx

    cublasDscal_v2 = cublasSscal_v2
    cublasCscal_v2 = cublasSscal_v2
    cublasZscal_v2 = cublasSscal_v2
    cublasCsscal_v2 = cublasSscal_v2
    cublasZdscal_v2 = cublasSscal_v2

    cublasSaxpy_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int,          # n
                                    c_void_p,       # alpha h/d
                                    c_void_p,       # x
                                    c_int,          # incx
                                    c_void_p,       # y
                                    c_int)          # incy
    cublasDaxpy_v2 = cublasSaxpy_v2
    cublasCaxpy_v2 = cublasSaxpy_v2
    cublasZaxpy_v2 = cublasSaxpy_v2

    cublasIsamax_v2 = ctype_function(cublasStatus_t,
                                     cublasHandle_t, # handle
                                     c_int,          # n
                                     c_void_p,       # x
                                     c_int,          # incx
                                     POINTER(c_int)) # result h/d ptr
    
    cublasIdamax_v2 = cublasIsamax_v2
    cublasIcamax_v2 = cublasIsamax_v2
    cublasIzamax_v2 = cublasIsamax_v2

    cublasIsamin_v2 = cublasIsamax_v2

    cublasIdamin_v2 = cublasIsamin_v2
    cublasIcamin_v2 = cublasIsamin_v2
    cublasIzamin_v2 = cublasIsamin_v2

    cublasSasum_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int,          # n
                                    c_void_p,       # x
                                    c_int,          # incx
                                    c_void_p)       # result h/d ptr

    cublasDasum_v2 = cublasSasum_v2
    cublasScasum_v2 = cublasSasum_v2
    cublasDzasum_v2 = cublasSasum_v2

    cublasSrot_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t,  # handle,
                                   c_int,           # n
                                   c_void_p,        # x
                                   c_int,           # incx
                                   c_void_p,        # y
                                   c_int,           # incy
                                   c_void_p,        # c
                                   c_void_p)        # s h/d ptr

    cublasDrot_v2 = cublasSrot_v2
    cublasCrot_v2 = cublasSrot_v2
    cublasZrot_v2 = cublasSrot_v2
    cublasCsrot_v2 = cublasSrot_v2
    cublasZdrot_v2 = cublasSrot_v2

    cublasSrotg_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t,     # handle,
                                    c_void_p,           # a h/d ptr
                                    c_void_p,           # b h/d ptr
                                    c_void_p,           # c h/d ptr
                                    c_void_p)           # s h/d ptr

    cublasDrotg_v2 = cublasSrotg_v2
    cublasCrotg_v2 = cublasSrotg_v2
    cublasZrotg_v2 = cublasSrotg_v2

    cublasSrotm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t,     # handle
                                    c_int,              # n
                                    c_void_p,           # x
                                    c_int,              # incx
                                    c_void_p,           # y
                                    c_int,              # incy
                                    c_void_p)           # param h/d pointer
    cublasDrotm_v2 = cublasSrotm_v2

    cublasSrotmg_v2 = ctype_function(cublasStatus_t,
                                     cublasHandle_t, # handle,
                                     c_void_p,       # d1 h/d ptr
                                     c_void_p,       # d2 h/d ptr
                                     c_void_p,       # x1 h/d ptr
                                     c_void_p,       # y1 h/d ptr
                                     c_void_p)       # param h/d ptr

    cublasDrotmg_v2 = cublasSrotmg_v2

class c_complex(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]

    def __new__(cls, real=0, imag=0):
        if isinstance(real, complex):
            real = real.real
            imag = real.imag
        return super(c_complex, cls).__new__(cls, real=real, imag=imag)

class c_double_complex(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]

    def __new__(cls, real=0, imag=0):
        if isinstance(real, complex):
            real = real.real
            imag = real.imag
        return super(c_double_complex, cls).__new__(cls, real=real, imag=imag)

def _return_scalar(result):
    if isinstance(result, (c_complex, c_double_complex)):
        return complex(result.real, result.imag)
    else:
        return result.value

def _Tnrm2(fmt, cty):
    def _wrapped(self, n, x, incx):
        result = cty()
        fn = getattr(self._api, 'cublas%snrm2_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           byref(result))
        return _return_scalar(result)
    return _wrapped

def _Tdot(fmt, cty, postfix=''):
    def _wrapped(self, n, x, incx, y, incy):
        result = cty()
        fn = getattr(self._api, 'cublas%sdot%s_v2' % (fmt, postfix))
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           y.device_raw_ptr.value, int(incy), byref(result))
        return _return_scalar(result)
    return _wrapped

def _Tscal(fmt, cty):
    def _wrapped(self, n, alpha, x, incx):
        "Stores result to x"
        c_alpha = cty(alpha)
        fn = getattr(self._api, 'cublas%sscal_v2' % fmt)
        fn(self._handle, int(n), byref(c_alpha), x.device_raw_ptr.value,
           int(incx))
    return _wrapped

def _Taxpy(fmt, cty):
    def _wrapped(self, n, alpha, x, incx, y, incy):
        "Stores result to y"
        c_alpha = cty(alpha)
        fn = getattr(self._api, 'cublas%saxpy_v2' % fmt)
        fn(self._handle, int(n), byref(c_alpha), x.device_raw_ptr.value,
           int(incx), y.device_raw_ptr.value, int(incy))
    return _wrapped

def _Itamax(fmt, cty):
    def _wrapped(self, n, x, incx):
        result = c_int()
        fn = getattr(self._api, 'cublasI%samax_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           byref(result))
        return result.value
    return _wrapped

def _Itamin(fmt, cty):
    def _wrapped(self, n, x, incx):
        result = c_int()
        fn = getattr(self._api, 'cublasI%samin_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           byref(result))
        return result.value
    return _wrapped

def _Tasum(fmt, cty):
    def _wrapped(self, n, x, incx):
        result = cty()
        fn = getattr(self._api, 'cublas%sasum_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           byref(result))
        return _return_scalar(result)
    return _wrapped

def _Trot(fmt, cty, sty):
    def _wrapped(self, n, x, incx, y, incy, c, s):
        "Stores to x and y"
        c_c = cty(c)
        c_s = sty(s)
        fn = getattr(self._api, 'cublas%srot_v2' % fmt)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           y.device_raw_ptr.value, int(incy), byref(c_c), byref(c_s))
    return _wrapped

def _Trotg(fmt, ty, cty):
    def _wrapped(self, a, b):
        c_a = ty(a)
        c_b = ty(b)
        c_c = cty()
        c_s = ty()
        fn = getattr(self._api, 'cublas%srotg_v2' % fmt)
        fn(self._handle, byref(c_a), byref(c_b), byref(c_c), byref(c_s))
        r, z, c, s = map(_return_scalar, [c_a, c_b, c_c, c_s])
        return r, z, c, s
    return _wrapped

def _Trotm(fmt, dtype):
    def _wrapped(self, n, x, incx, y, incy, param):
        "Stores result to x, y"
        fn = getattr(self._api, 'cublas%srotm_v2' % fmt)
        assert len(param.shape) == 1
        assert param.size >= 5
        assert param.dtype == np.dtype(dtype)
        fn(self._handle, int(n), x.device_raw_ptr.value, int(incx),
           y.device_raw_ptr.value, int(incy), param.ctypes.data)
    return _wrapped

def _Trotmg(fmt, cty, dtype):
    def _wrapped(self, d1, d2, x1, y1):
        fn = getattr(self._api, 'cublas%srotmg_v2' % fmt)
        c_d1 = cty(d1)
        c_d2 = cty(d2)
        c_x1 = cty(x1)
        c_y1 = cty(y1)
        param = np.zeros(5, dtype=dtype)
        fn(self._handle, byref(c_d1), byref(c_d2), byref(c_x1), byref(c_y1),
           param.ctypes.data)
        return param
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

    Sdot = _Tdot('S', c_float)
    Ddot = _Tdot('D', c_double)
    Cdotu = _Tdot('C', c_complex, 'u')
    Zdotu = _Tdot('Z', c_double_complex, 'u')
    Cdotc = _Tdot('C', c_complex, 'c')
    Zdotc = _Tdot('Z', c_double_complex, 'c')

    Sscal = _Tscal('S', c_float)
    Dscal = _Tscal('D', c_double)
    Cscal = _Tscal('C', c_complex)
    Zscal = _Tscal('Z', c_double_complex)
    Csscal = _Tscal('Cs', c_float)
    Zdscal = _Tscal('Zd', c_double)

    Saxpy = _Taxpy('S', c_float)
    Daxpy = _Taxpy('D', c_double)
    Caxpy = _Taxpy('C', c_complex)
    Zaxpy = _Taxpy('Z', c_double_complex)

    Isamax = _Itamax('s', c_float)
    Idamax = _Itamax('d', c_double)
    Icamax = _Itamax('c', c_complex)
    Izamax = _Itamax('z', c_double_complex)

    Isamin = _Itamin('s', c_float)
    Idamin = _Itamin('d', c_double)
    Icamin = _Itamin('c', c_complex)
    Izamin = _Itamin('z', c_double_complex)

    Sasum = _Tasum('S', c_float)
    Dasum = _Tasum('D', c_double)
    Scasum = _Tasum('Sc', c_float)
    Dzasum = _Tasum('Dz', c_double)

    Srot = _Trot('S', c_float, c_float)
    Drot = _Trot('D', c_double, c_double)
    Crot = _Trot('C', c_float, c_complex)
    Zrot = _Trot('Z', c_double, c_double_complex)
    Csrot = _Trot('Cs', c_float, c_float)
    Zdrot = _Trot('Zd', c_double, c_double)

    Srotg = _Trotg('S', c_float, c_float)
    Drotg = _Trotg('D', c_double, c_double)
    Crotg = _Trotg('C', c_complex, c_float)
    Zrotg = _Trotg('Z', c_double_complex, c_double)

    Srotm = _Trotm('S', np.float32)
    Drotm = _Trotm('D', np.float64)

    Srotmg = _Trotmg('S', c_float, np.float32)
    Drotmg = _Trotmg('D', c_double, np.float64)

