from contextlib import contextmanager
import numpy as np
from .binding import cuBlas
from numbapro import cuda

def _dtype_vtable(table):
    return dict((np.dtype(k), v) for k, v in table.iteritems())

def _sel_complex(real, imag):
    return {float: real,
            complex: imag,
            np.float32: real,
            np.float64: real,
            np.complex64: imag,
            np.complex128: imag,}


class Blas(object):
    @cuda.require_context
    def __init__(self, stream=0):
        self._cublas = cuBlas()
        if stream:
            self._cublas.stream = stream

    @property
    def stream(self):
        return self._cublas.stream

    @contextmanager
    def _auto(self, *arys):
        ctx = (cuda._auto_device(ary, stream=self.stream) for ary in arys)
        darys, convs = zip(*ctx)
        if len(darys) == 1:
            yield darys[0]
        else:
            yield darys
        for dary, conv in zip(darys, convs):
            if conv:
                dary.to_host(stream=self.stream)

    @contextmanager
    def _auto_read(self, *arys):
        ctx = (cuda._auto_device(ary, stream=self.stream) for ary in arys)
        darys, convs = zip(*ctx)
        if len(darys) == 1:
            yield darys[0]
        else:
            yield darys

    def _dispatch(self, vtable, *keys):
        rsvl = vtable
        for k in keys:
            if not isinstance(rsvl, dict):
                break

            try:
                rsvl = rsvl[k]
            except KeyError:
                raise TypeError(k)
    
        return getattr(self._cublas, rsvl)

    def nrm2(self, x):
        "Same as np.linalg.norm"
        _sentry_same_dtype(x)
        _sentry_ndim(1, x)
        fn = self._dispatch(self.nrm2.vtable, x.dtype)
        with self._auto_read(x) as dx:
            return fn(x.size, dx, *_norm_stride(x))

    nrm2.vtable = _dtype_vtable({np.float32:    'Snrm2',
                                 np.float64:    'Dnrm2',
                                 np.complex64:  'Scnrm2',
                                 np.complex128: 'Dznrm2'})

    def dot(self, x, y):
        "Same as np.dot"
        _sentry_same_dtype(x, y)
        _sentry_ndim(1, x, y)
        _sentry_same_shape(x, y)
        fn = self._dispatch(self.dot.vtable, x.dtype)
        with self._auto_read(x, y) as (dx, dy):
            return fn(x.size, dx, _norm_stride(x)[0], dy, _norm_stride(y)[0])

    dot.vtable = _dtype_vtable({np.float32:    'Sdot',
                                np.float64:    'Ddot',
                                np.complex64:  'Cdotu',
                                np.complex128: 'Zdotu',})

    def dotu(self, x, y):
        "Alias to dot for complex type."
        if x.dtype not in map(np.dtype, [np.complex64, np.complex128]):
            raise TypeError("Invalid dtype for dotu")
        return self.dot(x, y)

    def dotc(self, x, y):
        "Same as np.vdot"
        _sentry_same_dtype(x, y)
        _sentry_ndim(1, x, y)
        _sentry_same_shape(x, y)
        fn = self._dispatch(self.dotc.vtable, x.dtype)
        with self._auto_read(x, y) as (dx, dy):
            return fn(x.size, dx, _norm_stride(x)[0], dy, _norm_stride(y)[0])

    dotc.vtable = _dtype_vtable({np.complex64:  'Cdotc',
                                 np.complex128: 'Zdotc',})

    def scal(self, alpha, x):
        "Same as x = alpha * x"
        _sentry_ndim(1, x)
        fn = self._dispatch(self.scal.vtable, x.dtype, type(alpha))
        with self._auto(x) as dx:
            return fn(x.size, alpha, dx, *_norm_stride(x))

    scal.vtable = _dtype_vtable({np.float32:    'Sscal',
                                 np.float64:    'Dscal',
                                 np.complex64:  _sel_complex(imag='Cscal',
                                                             real='Csscal'),
                                 np.complex128: _sel_complex(imag='Zscal',
                                                             real='Zdscal')})
    def axpy(self, alpha, x, y):
        "Same as y = alpha * x + y"
        _sentry_ndim(1, x, y)
        _sentry_same_dtype(x, y)
        _sentry_same_shape(x, y)
        fn = self._dispatch(self.axpy.vtable, x.dtype)
        with self._auto_read(x) as dx:
            with self._auto(y) as dy:
                return fn(x.size, alpha, dx, _norm_stride(x)[0], dy,
                          _norm_stride(y)[0])

    axpy.vtable = _dtype_vtable({np.float32:    'Saxpy',
                                 np.float64:    'Daxpy',
                                 np.complex64:  'Caxpy',
                                 np.complex128: 'Zaxpy'})

    def amax(self, x):
        "Same as np.argmax(x)"
        _sentry_ndim(1, x)
        fn = self._dispatch(self.amax.vtable, x.dtype)
        with self._auto_read(x) as dx:
            return fn(x.size, dx, _norm_stride(x)[0]) - 1

    amax.vtable = _dtype_vtable({np.float32:    'Isamax',
                                 np.float64:    'Idamax',
                                 np.complex64:  'Icamax',
                                 np.complex128: 'Izamax'})

    def amin(self, x):
        "Same as np.argmin(x)"
        _sentry_ndim(1, x)
        fn = self._dispatch(self.amin.vtable, x.dtype)
        with self._auto_read(x) as dx:
            return fn(x.size, dx, _norm_stride(x)[0]) - 1

    amin.vtable = _dtype_vtable({np.float32:    'Isamin',
                                 np.float64:    'Idamin',
                                 np.complex64:  'Icamin',
                                 np.complex128: 'Izamin'})

    def asum(self, x):
        "Same as np.sum(x)"
        _sentry_ndim(1, x)
        fn = self._dispatch(self.asum.vtable, x.dtype)
        with self._auto_read(x) as dx:
            return fn(x.size, dx, _norm_stride(x)[0])

    asum.vtable = _dtype_vtable({np.float32:    'Sasum',
                                 np.float64:    'Dasum',
                                 np.complex64:  'Scasum',
                                 np.complex128: 'Dzasum'})

    def rot(self, x, y, c, s):
        "Same as x, y = c * x + s * y, -s * x + c * y"
        _sentry_ndim(1, x, y)
        fn = self._dispatch(self.rot.vtable, x.dtype, type(s))
        with self._auto(x, y) as (dx, dy):
            return fn(x.size, dx, _norm_stride(x)[0], dy, _norm_stride(x)[0],
                      c, s)

    rot.vtable = _dtype_vtable({np.float32:    'Srot',
                                np.float64:    'Drot',
                                np.complex64:  _sel_complex(imag='Crot',
                                                            real='Csrot'),
                                np.complex128: _sel_complex(imag='Zrot',
                                                            real='Zdrot')})

    def rotg(self, a, b):
        '''Compute the given rotation matrix given a column vector (a, b).
        Returns r, z, c, s.
        
        r --- r = a ** 2 + b ** 2
        z --- Use to recover c and s.
              if abs(z) < 1: 
                  c, s = 1 - z ** 2, z
              elif abs(z) == 1:
                  c, s = 0, 1
              else:
                  c, s = 1 / z, 1 - z ** 2
        c --- Cosine element of the rotation matrix.
        s --- Sine element of the rotation matrix.
        '''
        a, b = np.asarray(a), np.asarray(b)
        _sentry_same_dtype(a, b)
        fn = self._dispatch(self.rotg.vtable, a.dtype)
        return fn(np.asscalar(a), np.asscalar(b))

    rotg.vtable = _dtype_vtable({np.float32:        'Srotg',
                                 np.float64:        'Drotg',
                                 np.complex64:      'Crotg',
                                 np.complex128:     'Zrotg'})

    def rotm(self, x, y, param):
        '''Applies the modified Givens transformation.

        x, y = h11 * x + h12 * y, h21 * x + h22 * y
        
        param --- [flag, h11, h21, h12, h22]

        Refer to cuBLAS documentation for detail.
        '''
        _sentry_ndim(1, x, y)
        _sentry_same_dtype(x, y)
        _sentry_same_shape(x, y)
        fn = self._dispatch(self.rotm.vtable, x.dtype)
        with self._auto(x, y) as (dx, dy):
            return fn(x.size, dx, _norm_stride(x)[0], dy, _norm_stride(y)[0],
                      param)

    rotm.vtable = _dtype_vtable({np.float32:       'Srotm',
                                 np.float64:       'Drotm'})

    def rotmg(self, d1, d2, x1, y1):
        '''Constructs the modified Givens transformation.
        
        Returns param that is usable in rotm.
        
        Refer to cuBLAS documentation for detail.
        '''
        d1, d2, x1, y1 = map(np.asarray, [d1, d2, x1, y1])
        _sentry_same_dtype(d1, d2, x1, y1)
        fn = self._dispatch(self.rotmg.vtable, x1.dtype)
        return fn(*map(np.asscalar, [d1, d2, x1, y1]))

    rotmg.vtable = _dtype_vtable({np.float32: 'Srotmg',
                                  np.float64: 'Drotmg'})


#----------------
# utils
#----------------

def _sentry_same_shape(*arys):
    first = arys[0]
    for ary in arys:
        if ary.shape != first.shape:
            raise TypeError("Expecting all arrays to have the same shape.")

def _sentry_same_dtype(*arys):
    first = arys[0]
    for ary in arys:
        if ary.dtype != first.dtype:
            raise TypeError("All arrays must have the same dtype.")

def _sentry_ndim(ndim, *arys):
    for ary in arys:
        if ary.ndim != ndim:
            raise TypeError("Expecting %d dimension array." % ndim)

def _norm_stride(ary):
    retval = []
    for stride in ary.strides:
        if stride % ary.dtype.itemsize != 0:
            raise ValueError("Misalignment.")
        retval.append(stride // ary.dtype.itemsize)
    return retval
