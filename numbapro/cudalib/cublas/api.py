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

def _auto_l2_functions(fname, tnames, argfmt, extras):
    writebacks = set()
    readonlys = set()
    arglist = []
    extras = map(lambda s: s.lstrip().rstrip(), extras.split(','))
    dtypemap = {
        np.dtype(np.float32): 'S',
        np.dtype(np.float64): 'D',
        np.dtype(np.complex64): 'C',
        np.dtype(np.complex128): 'Z',
    }
    for i, a in enumerate(argfmt.split(',')):
        a = a.lstrip().rstrip()
        if ':' in a:
            name, mode = a.split(':')
            assert mode in 'wr'
            if mode == 'w':
                writebacks.add(name)
            else:
                readonlys.add(name)
        else:
            name = a
        arglist.append(name)

    def prepare_args(args, kws):
        for i, a in enumerate(args):
            name = arglist[i]
            assert name not in kws
            kws[name] = a
        for a in extras:
            if a.startswith('ld') and len(a) == 3:
                kws[a] = kws[a[-1].upper()].shape[0]
            elif a.startswith('inc') and len(a) == 4:
                ary = kws[a[-1]]
                kws[a] = ary.strides[0] / ary.dtype.itemsize
            else:
                assert False, 'unreachable'

    devargs = list(writebacks | readonlys)

    def autodevice(kws, stream):
        newkws = kws.copy()
        cleanups = []
        for a in readonlys:
            newkws[a], _ = cuda._auto_device(kws[a], stream=stream)
        for a in writebacks:
            dmem, conv = cuda._auto_device(kws[a], stream=stream)
            newkws[a] = dmem
            if conv:
                cleanups.append(dmem)
        return newkws, cleanups

    def _dispatch(self, *args, **kws):
        prepare_args(args, kws)
        dtype = kws[devargs[0]].dtype
        typecode = dtypemap[dtype]
        assert typecode in tnames
        fn = getattr(self._cublas, '%s%s' % (typecode, fname))
        kws, cleanups = autodevice(kws, self.stream)
        res = fn(**kws)
        for dmem in cleanups:
            dmem.to_host(stream=self.stream)
        return res

    return _dispatch

class Blas(object):
    '''All BLAS subprograms are available under the Blas object.
    
    :param stream: Optional. A CUDA Stream.
    '''
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

    # Level 2

    gbmv = _auto_l2_functions('gbmv', 'SDCZ',
                             'trans, m, n, kl, ku, alpha, A:r, x:r, beta, y:w',
                             'lda, incx, incy')

    gemv = _auto_l2_functions('gemv', 'SDCZ',
                              'trans, m, n, alpha, A:r, x:r, beta, y:w',
                              'lda, incx, incy')

    trmv = _auto_l2_functions('trmv', 'SDCZ',
                              'uplo, trans, diag, n, A:r, x:w',
                              'lda, incx')
    
    tbmv = _auto_l2_functions('tbmv', 'SDCZ',
                              'uplo, trans, diag, n, k, A:r, x:w',
                              'lda, incx')

    tpmv = _auto_l2_functions('tpmv', 'SDCZ',
                              'uplo, trans, diag, n, AP:r, x:w',
                              'incx')

    trsv = _auto_l2_functions('trsv', 'SDCZ',
                              'uplo, trans, diag, n, A:r, x:w',
                              'lda, incx')

    tpsv = _auto_l2_functions('tpsv', 'SDCZ',
                              'uplo, trans, diag, n, AP:r, x:w',
                              'incx')

    tbsv = _auto_l2_functions('tbsv', 'SDCZ',
                              'uplo, trans, diag, n, k, A:r, x:w',
                              'lda, incx')

    symv = _auto_l2_functions('symv', 'SDCZ',
                              'uplo, n, alpha, A:r, x:r, beta, y:w',
                              'lda, incx, incy')

    hemv = _auto_l2_functions('hemv', 'CZ',
                              'uplo, n, alpha, A:r, x:r, beta, y:w',
                              'lda, incx, incy')

    sbmv = _auto_l2_functions('sbmv', 'SDCZ',
                              'uplo, n, k, alpha, A:r, x:r, beta, y:w',
                              'lda, incx, incy')
    
    hbmv = _auto_l2_functions('hbmv', 'CZ',
                              'uplo, n, k, alpha, A:r, x:r, beta, y:w',
                              'lda, incx, incy')

    spmv = _auto_l2_functions('spmv', 'SD',
                              'uplo, n, alpha, AP:r, x:r, beta, y:w',
                              'incx, incy')

    hpmv = _auto_l2_functions('hpmv', 'CZ',
                              'uplo, n, alpha, AP:r, x:r, beta, y:w',
                              'incx, incy')

    ger = _auto_l2_functions('ger', 'SD',
                             'm, n, alpha, x:r, y:r, A:w',
                             'incx, incy, lda')

    geru = _auto_l2_functions('geru', 'CZ',
                              'm, n, alpha, x:r, y:r, A:w',
                              'incx, incy, lda')

    gerc = _auto_l2_functions('gerc', 'CZ',
                              'm, n, alpha, x:r, y:r, A:w',
                              'incx, incy, lda')

    syr = _auto_l2_functions('syr', 'SDCZ', 'uplo, n, alpha, x:r, A:w',
                             'incx, lda')

    her = _auto_l2_functions('her', 'CZ', 'uplo, n, alpha, x:r, A:w',
                             'incx, lda')

    spr = _auto_l2_functions('spr', 'SD', 'uplo, n, alpha, x:r, AP:w',
                             'incx')

    hpr = _auto_l2_functions('hpr', 'CZ', 'uplo, n, alpha, x:r, AP:w',
                             'incx')

    syr2 = _auto_l2_functions('syr2', 'SDCZ',
                              'uplo, n, alpha, x:r, y:r, A:w',
                              'incx, incy, lda')
    
    her2 = _auto_l2_functions('her2', 'CZ',
                              'uplo, n, alpha, x:r, y:r, A:w',
                              'incx, incy, lda')

    spr2 = _auto_l2_functions('spr2', 'SDCZ',
                              'uplo, n, alpha, x:r, y:r, A:w',
                              'incx, incy')

    hpr2 = _auto_l2_functions('hpr2', 'CZ',
                              'uplo, n, alpha, x:r, y:r, A:w',
                              'incx, incy')

    # Level 3

    gemm = _auto_l2_functions('gemm', 'SDCZ',
                          'transa, transb, m, n, k, alpha, A:r, B:r, beta, C:w',
                          'lda, ldb, ldc')

    syrk = _auto_l2_functions('syrk', 'SDCZ',
                              'uplo, trans, n, k, alpha, A:r, beta, C:w',
                              'lda, ldc')

    herk = _auto_l2_functions('herk', 'CZ',
                              'uplo, trans, n, k, alpha, A:r, beta, C:w',
                              'lda, ldc')
    
    symm = _auto_l2_functions('symm', 'SDCZ',
                              'side, uplo, m, n, alpha, A:r, B:r, beta, C:w',
                              'lda, ldb, ldc')
    
    hemm = _auto_l2_functions('hemm', 'CZ',
                              'side, uplo, m, n, alpha, A:r, B:r, beta, C:w',
                              'lda, ldb, ldc')

    trsm = _auto_l2_functions('trsm', 'SDCZ',
                              'side, uplo, trans, diag, m, n, alpha, A:r, B:w',
                              'lda, ldb')

    trmm = _auto_l2_functions('trmm', 'SDCZ',
                          'side, uplo, trans, diag, m, n, alpha, A:r, B:r, C:w',
                          'lda, ldb, ldc')

    dgmm = _auto_l2_functions('dgmm', 'SDCZ',
                              'side, m, n, A:r, x:r, C:w',
                              'lda, ldc, incx')

    geam = _auto_l2_functions('geam', 'SDCZ',
                            'transa, transb, m, n, alpha, A:r, beta, B:r, C:w',
                            'lda, ldb, ldc')

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
