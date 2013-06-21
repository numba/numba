import numpy as np
import unittest
from .support import addtest

from numbapro import cuda

@addtest
class TestCuBlasBinding(unittest.TestCase):
    def test_lib(self):
        from numbapro.cudalib.cublas.binding import (cuBlas,
                          CUBLAS_POINTER_MODE_HOST, CUBLAS_ATOMICS_NOT_ALLOWED)
        stream = cuda.stream()
        blas = cuBlas()
        print blas.version
        blas.stream = stream
        self.assertTrue(blas.stream is stream)
        blas.pointer_mode = CUBLAS_POINTER_MODE_HOST
        self.assertTrue(blas.pointer_mode == CUBLAS_POINTER_MODE_HOST)
        blas.atomics_mode = CUBLAS_ATOMICS_NOT_ALLOWED
        self.assertTrue(blas.atomics_mode == CUBLAS_ATOMICS_NOT_ALLOWED)

    def Tnrm2(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)
        d_x = cuda.to_device(x)

        blas = cuBlas()
        got = getattr(blas, fn)(x.size, d_x, 1)
        exp = np.linalg.norm(x)
        self.assertTrue(np.allclose(got, exp))

    def test_Snrm2(self):
        self.Tnrm2('Snrm2', np.float32)

    def test_Dnrm2(self):
        self.Tnrm2('Dnrm2', np.float64)

    def test_Scnrm2(self):
        self.Tnrm2('Scnrm2', np.complex64)

    def test_Dznrm2(self):
        self.Tnrm2('Dznrm2', np.complex128)

    def Tdot(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)

        blas = cuBlas()
        got = getattr(blas, fn)(x.size, d_x, 1, d_y, 1)
        if fn.endswith('c'):
            exp = np.vdot(x, y)
        else:
            exp = np.dot(x, y)
        self.assertTrue(np.allclose(got, exp))

    def test_Sdot(self):
        self.Tdot('Sdot', np.float32)

    def test_Ddot(self):
        self.Tdot('Ddot', np.float64)

    def test_Cdotu(self):
        self.Tdot('Cdotu', np.complex64)

    def test_Zdotu(self):
        self.Tdot('Zdotu', np.complex128)

    def test_Cdotu(self):
        self.Tdot('Cdotc', np.complex64)

    def test_Zdotu(self):
        self.Tdot('Zdotc', np.complex128)

    def Tscal(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        alpha = 1.234
        x = np.random.random(10).astype(dtype)
        x0 = x.copy()
        d_x = cuda.to_device(x)

        blas = cuBlas()
        getattr(blas, fn)(x.size, alpha, d_x, 1)

        d_x.to_host()

        self.assertTrue(np.allclose(x0 * alpha, x))

    def test_Sscal(self):
        self.Tscal('Sscal', np.float32)

    def test_Dscal(self):
        self.Tscal('Dscal', np.float64)

    def test_Cscal(self):
        self.Tscal('Cscal', np.complex64)

    def test_Zscal(self):
        self.Tscal('Zscal', np.complex128)

    def test_Csscal(self):
        self.Tscal('Csscal', np.complex64)

    def test_Zdscal(self):
        self.Tscal('Zdscal', np.complex128)

    def Taxpy(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        alpha = 1.234
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        y0 = y.copy()

        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)

        blas = cuBlas()
        getattr(blas, fn)(x.size, alpha, d_x, 1, d_y, 1)

        d_y.to_host()

        self.assertTrue(np.allclose(alpha * x + y0, y))

    def test_Saxpy(self):
        self.Taxpy('Saxpy', np.float32)

    def test_Daxpy(self):
        self.Taxpy('Daxpy', np.float64)

    def test_Caxpy(self):
        self.Taxpy('Caxpy', np.complex64)

    def test_Zaxpy(self):
        self.Taxpy('Zaxpy', np.complex128)

    def Itamax(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)

        d_x = cuda.to_device(x)

        blas = cuBlas()
        got = getattr(blas, fn)(x.size, d_x, 1) - 1
        self.assertTrue(np.allclose(np.argmax(x), got))

    def test_Isamax(self):
        self.Itamax('Isamax', np.float32)

    def test_Idamax(self):
        self.Itamax('Idamax', np.float64)

    def test_Icamax(self):
        self.Itamax('Icamax', np.complex64)

    def test_Izamax(self):
        self.Itamax('Izamax', np.complex128)

    def Itamin(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)

        d_x = cuda.to_device(x)

        blas = cuBlas()
        got = getattr(blas, fn)(x.size, d_x, 1) - 1
        self.assertTrue(np.allclose(np.argmin(x), got))

    def test_Isamin(self):
        self.Itamin('Isamin', np.float32)

    def test_Idamin(self):
        self.Itamin('Idamin', np.float64)

    def test_Icamin(self):
        self.Itamin('Icamin', np.complex64)

    def test_Izamin(self):
        self.Itamin('Izamin', np.complex128)

    def Tasum(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)

        d_x = cuda.to_device(x)

        blas = cuBlas()
        got = getattr(blas, fn)(x.size, d_x, 1)
        self.assertTrue(np.allclose(np.sum(x), got))

    def test_Sasum(self):
        self.Tasum('Sasum', np.float32)

    def test_Dasum(self):
        self.Tasum('Dasum', np.float64)

    def test_Scasum(self):
        self.Tasum('Scasum', np.complex64)

    def test_Dzasum(self):
        self.Tasum('Dzasum', np.complex128)

    def Trot(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        angle = 1.342
        c = np.cos(angle)
        s = np.sin(angle)

        x0, y0 = c * x + s * y, -s * x + c * y

        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)

        blas = cuBlas()
        getattr(blas, fn)(x.size, d_x, 1, d_y, 1, c, s)

        d_x.to_host()
        d_y.to_host()

        self.assertTrue(np.allclose(x, x0))
        self.assertTrue(np.allclose(y, y0))
    
    def test_Srot(self):
        self.Trot('Srot', np.float32)

    def test_Drot(self):
        self.Trot('Drot', np.float64)

    def test_Crot(self):
        self.Trot('Crot', np.complex64)

    def test_Zrot(self):
        self.Trot('Zrot', np.complex128)

    def test_Csrot(self):
        self.Trot('Csrot', np.complex64)

    def test_Zdrot(self):
        self.Trot('Zdrot', np.complex128)

    def Trotg(self, fn):
        from numbapro.cudalib.cublas.binding import cuBlas
        a, b = np.random.random(), np.random.random()
        blas = cuBlas()
        r, z, c, s = getattr(blas, fn)(a, b)

        rot = np.array([[c,           s],
                        [-np.conj(s), c]])
        vec = np.array([[a],
                        [b]])
        exp = np.dot(rot, vec)
        got = np.array([[r],
                        [0.0]])
        self.assertTrue(np.allclose(exp, got, atol=1e-6))

    def test_Srotg(self):
        self.Trotg('Srotg')

    def test_Drotg(self):
        self.Trotg('Drotg')

    def test_Crotg(self):
        self.Trotg('Crotg')

    def test_Zrotg(self):
        self.Trotg('Zrotg')

    def Trotm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)

        param = np.random.random(5).astype(dtype)
        param[0] = -1.0
        h11, h21, h12, h22 = param[1:].tolist()

        x0, y0 = h11 * x + h12 * y, h21 * x + h22 * y

        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)

        blas = cuBlas()
        getattr(blas, fn)(x.size, d_x, 1, d_y, 1, param)

        d_x.to_host()
        d_y.to_host()

        self.assertTrue(np.allclose(x, x0))
        self.assertTrue(np.allclose(y, y0))

    def test_Srotm(self):
        self.Trotm('Srotm', np.float32)

    def test_Drotm(self):
        self.Trotm('Drotm', np.float64)

    def Trotmg(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        d1, d2, x1, y1 = np.random.random(4).tolist()
        
        blas = cuBlas()
        param = getattr(blas, fn)(d1, d2, x1, y1)

        flag, h11, h21, h12, h22 = param.tolist()

        if flag == -1.0:
            pass # don't know how to check
        elif flag == 0.0:
            self.assertEqual(h11, 0)
            self.assertEqual(h22, 0)
        elif flag == 1.0:
            self.assertEqual(h12, 0)
            self.assertEqual(h21, 0)
        else:
            self.assertEqual(flag, -2.0)
            self.assertEqual(h11, 0)
            self.assertEqual(h12, 0)
            self.assertEqual(h21, 0)
            self.assertEqual(h22, 0)

    def test_Srotmg(self):
        self.Trotmg('Srotmg', np.float32)

    def test_Drotmg(self):
        self.Trotmg('Drotmg', np.float64)

    #
    # Level 2 tests
    # They just simply test to see if the binding works; doesn't check for
    # correct result.
    #

    def Tgbmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        blas = cuBlas()
        kl = 0
        ku = 0
        alpha = 1.
        beta = 0.
        A = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 3]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([1, 2, 3], dtype=dtype)
        lda, n = A.shape
        m = lda
        y0 = y.copy()
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)
        getattr(blas, fn)('N', m, n, kl, ku, alpha, dA, lda, dx, 1, beta, dy, 1)
        dy.copy_to_host(y)
        self.assertFalse(all(y0 == y))

    def test_Sgbmv(self):
        self.Tgbmv('Sgbmv', np.float32)

    def test_Dgbmv(self):
        self.Tgbmv('Dgbmv', np.float64)

    def test_Cgbmv(self):
        self.Tgbmv('Cgbmv', np.complex64)

    def test_Zgbmv(self):
        self.Tgbmv('Zgbmv', np.complex128)

    def Tgemv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        blas = cuBlas()
        kl = 0
        ku = 0
        alpha = 1.
        beta = 0.
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([1, 2, 3], dtype=dtype)
        m, n = A.shape
        lda = m
        y0 = y.copy()
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)
        getattr(blas, fn)('N', m, n, alpha, dA, lda, dx, 1, beta, dy, 1)
        dy.copy_to_host(y)
        self.assertFalse(all(y0 == y))

    def test_Sgemv(self):
        self.Tgemv('Sgemv', np.float32)

    def test_Dgemv(self):
        self.Tgemv('Dgemv', np.float64)

    def test_Cgemv(self):
        self.Tgemv('Cgemv', np.complex64)

    def test_Zgemv(self):
        self.Tgemv('Zgemv', np.complex128)

    def Ttrmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = True
        n = 3
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        lda = n
        x0 = x.copy()
        inc = 1
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        getattr(blas, fn)(uplo, trans, diag, n, dA, lda, dx, inc)
        dx.copy_to_host(x)
        self.assertFalse(all(x == x0))

    def test_Strmv(self):
        self.Ttrmv('Strmv', np.float32)

    def test_Dtrmv(self):
        self.Ttrmv('Dtrmv', np.float64)

    def test_Ctrmv(self):
        self.Ttrmv('Ctrmv', np.complex64)

    def test_Ztrmv(self):
        self.Ttrmv('Ztrmv', np.complex128)

    def Ttbmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        lda = n
        x0 = x.copy()
        inc = 1
        k = 0
        getattr(blas, fn)(uplo, trans, diag, n, k, dA, lda, dx, inc)
        dx.copy_to_host(x)
        
        self.assertFalse(all(x == x0))

    def test_Stbmv(self):
        self.Ttbmv('Stbmv', np.float32)

    def test_Dtbmv(self):
        self.Ttbmv('Dtbmv', np.float64)

    def test_Ctbmv(self):
        self.Ttbmv('Ctbmv', np.complex64)

    def test_Ztbmv(self):
        self.Ttbmv('Ztbmv', np.complex128)

    def Ttpmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        dAP = cuda.to_device(AP)
        dx = cuda.to_device(x)

        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        inc = 1
        getattr(blas, fn)(uplo, trans, diag, n, dAP, dx, inc)
        dx.copy_to_host(x)
        
        self.assertFalse(all(x == x0))

    def test_Stpmv(self):
        self.Ttpmv('Stpmv', np.float32)

    def test_Dtpmv(self):
        self.Ttpmv('Dtpmv', np.float64)

    def test_Ctpmv(self):
        self.Ttpmv('Ctpmv', np.complex64)

    def test_Ztpmv(self):
        self.Ttpmv('Ztpmv', np.complex128)

    def Ttrsv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = False
        lda = n = 3
        x0 = x.copy()
        inc = 1
        getattr(blas, fn)(uplo, trans, diag, n, dA, lda, dx, inc)
        dx.copy_to_host(x)
        
        self.assertFalse(all(x == x0))

    def test_Strsv(self):
        self.Ttrsv('Strsv', np.float32)

    def test_Dtrsv(self):
        self.Ttrsv('Dtrsv', np.float64)

    def test_Ctrsv(self):
        self.Ttrsv('Ctrsv', np.complex64)

    def test_Ztrsv(self):
        self.Ttrsv('Ztrsv', np.complex128)
    
    def _Ttpsv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        inc = 1
        getattr(blas, fn)(uplo, trans, diag, n, dA, dx, inc)
        dx.copy_to_host(x)
        
        self.assertFalse(all(x == x0))

    def test_Stpsv(self):
        self._Ttpsv('Stpsv', np.float32)

    def test_Dtpsv(self):
        self._Ttpsv('Dtpsv', np.float64)

    def test_Ctpsv(self):
        self._Ttpsv('Ctpsv', np.complex64)

    def test_Ztpsv(self):
        self._Ttpsv('Ztpsv', np.complex128)

    def _Ttbsv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        blas = cuBlas()
        uplo = 'U'
        trans = 'N'
        diag = False
        lda = n = 3
        k = 0
        x0 = x.copy()
        inc = 1
        getattr(blas, fn)(uplo, trans, diag, n, k, dA, lda, dx, inc)
        dx.copy_to_host(x)
        
        self.assertFalse(all(x == x0))

    def test_Stbsv(self):
        self._Ttbsv('Stbsv', np.float32)

    def test_Dtbsv(self):
        self._Ttbsv('Dtbsv', np.float64)

    def test_Ctbsv(self):
        self._Ttbsv('Ctbsv', np.complex64)

    def test_Ztbsv(self):
        self._Ttbsv('Ztbsv', np.complex128)

    def _Tsymv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2
        beta = .34
        blas = cuBlas()
        uplo = 'U'
        lda = n = 3
        k = 0
        y0 = y.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dA, lda, dx, incx, beta, dy, incy)
        dy.copy_to_host(y)
        
        self.assertFalse(all(y == y0))

    def test_Ssymv(self):
        self._Tsymv('Ssymv', np.float32)

    def test_Dsymv(self):
        self._Tsymv('Dsymv', np.float32)

    def test_Csymv(self):
        self._Tsymv('Csymv', np.float32)

    def test_Zsymv(self):
        self._Tsymv('Zsymv', np.float32)

    _Themv = _Tsymv

    def test_Chemv(self):
        self._Themv('Chemv', np.complex64)

    def test_Zhemv(self):
        self._Themv('Zhemv', np.complex128)

    def _Tsbmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2
        beta = .34
        blas = cuBlas()
        uplo = 'U'
        lda = n = 3
        k = 0
        y0 = y.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, k, alpha, dA, lda, dx, incx, beta, dy, incy)
        dy.copy_to_host(y)
        
        self.assertFalse(all(y == y0))

    def test_Ssbmv(self):
        self._Tsbmv('Ssbmv', np.float32)

    def test_Dsymv(self):
        self._Tsbmv('Dsbmv', np.float64)

    _Thbmv = _Tsbmv

    def test_Chbmv(self):
        self._Thbmv('Chbmv', np.complex64)

    def test_Zhbmv(self):
        self._Thbmv('Zhbmv', np.complex128)

    def _Tspmv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        dAP = cuda.to_device(AP)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2
        beta = .34
        blas = cuBlas()
        uplo = 'U'
        lda = n = 3
        k = 0
        y0 = y.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dAP, dx, incx, beta, dy, incy)
        dy.copy_to_host(y)
        
        self.assertFalse(all(y == y0))

    def test_Sspmv(self):
        self._Tspmv('Sspmv', np.float32)

    def test_Dspmv(self):
        self._Tspmv('Dspmv', np.float64)

    _Thpmv = _Tspmv

    def test_Chpmv(self):
        self._Thpmv('Chpmv', np.complex64)

    def test_Dspmv(self):
        self._Tspmv('Zhpmv', np.complex128)

    def _Tger(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2

        blas = cuBlas()

        lda = m = n = 3
        A0 = A.copy()
        incx = incy = 1
        getattr(blas, fn)(m, n, alpha, dx, incx, dy, incy, dA, lda)
        dA.copy_to_host(A)
        
        self.assertFalse(np.all(A == A0))

    def test_Sger(self):
        self._Tger('Sger', np.float32)

    def test_Dger(self):
        self._Tger('Dger', np.float64)

    def test_Cgeru(self):
        self._Tger('Cgeru', np.complex64)

    def test_Cgerc(self):
        self._Tger('Cgerc', np.complex64)

    def test_Zgeru(self):
        self._Tger('Zgeru', np.complex128)

    def test_Zgerc(self):
        self._Tger('Zgerc', np.complex128)


    def _Tsyr(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)

        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        alpha = 1.2
        uplo = 'U'
        blas = cuBlas()

        lda = m = n = 3
        A0 = A.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dx, incx, dA, lda)
        dA.copy_to_host(A)
        
        self.assertFalse(np.all(A == A0))

    def test_Ssyr(self):
        self._Tsyr('Ssyr', np.float32)

    def test_Dsyr(self):
        self._Tsyr('Dsyr', np.float64)

    def test_Csyr(self):
        self._Tsyr('Csyr', np.complex64)

    def test_Zsyr(self):
        self._Tsyr('Zsyr', np.complex128)

    def _Ther(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)

        dA = cuda.to_device(A)
        dx = cuda.to_device(x)

        alpha = 1.2
        uplo = 'U'
        blas = cuBlas()

        lda = m = n = 3
        A0 = A.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dx, incx, dA, lda)
        dA.copy_to_host(A)
        
        self.assertFalse(np.all(A == A0))

    def test_Cher(self):
        self._Ther('Cher', np.complex64)

    def test_Zher(self):
        self._Ther('Zher', np.complex128)

    def _Tspr(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)

        dAP = cuda.to_device(AP)
        dx = cuda.to_device(x)

        alpha = 1.2
        uplo = 'U'
        blas = cuBlas()

        lda = m = n = 3
        AP0 = AP.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dx, incx, dAP)
        dAP.copy_to_host(AP)
        
        self.assertFalse(np.all(AP == AP0))

    def test_Sspr(self):
        self._Tspr('Sspr', np.float32)

    def test_Dspr(self):
        self._Tspr('Dspr', np.float64)

    def test_Chpr(self):
        self._Tspr('Chpr', np.complex64)

    def test_Zhpr(self):
        self._Tspr('Zhpr', np.complex128)

    def _Tsyr2(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)

        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2
        uplo = 'U'
        blas = cuBlas()

        lda = m = n = 3
        A0 = A.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dx, incx, dy, incy, dA, lda)
        dA.copy_to_host(A)
        
        self.assertFalse(np.all(A == A0))

    _Ther2 = _Tsyr2

    def test_Ssyr2(self):
        self._Tsyr2('Ssyr2', np.float32)

    def test_Dsyr2(self):
        self._Tsyr2('Dsyr2', np.float64)

    def test_Csyr2(self):
        self._Tsyr2('Csyr2', np.complex64)

    def test_Zsyr2(self):
        self._Tsyr2('Zsyr2', np.complex128)

    def test_Cher2(self):
        self._Ther2('Cher2', np.complex64)

    def test_Zher2(self):
        self._Ther2('Zher2', np.complex128)

    def _Tspr2(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)

        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        alpha = 1.2
        uplo = 'U'
        blas = cuBlas()

        n = 3
        A0 = A.copy()
        incx = incy = 1
        getattr(blas, fn)(uplo, n, alpha, dx, incx, dy, incy, dA)
        dA.copy_to_host(A)
        
        self.assertFalse(np.all(A == A0))

    _Thpr2 = _Tspr2

    def test_Sspr2(self):
        self._Tspr2('Sspr2', np.float32)

    def test_Dspr2(self):
        self._Tspr2('Sspr2', np.float64)

    def test_Chpr2(self):
        self._Thpr2('Chpr2', np.complex64)

    def test_Zhpr2(self):
        self._Thpr2('Zhpr2', np.complex128)

    # Level 3

    def _Tgemm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)

        alpha = 1.2
        beta = .34

        transa = 'N'
        transb = 'N'
        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        getattr(blas, fn)(transa, transb, m, n, k, alpha, dA, lda, dB, ldb,
                          beta, dC, ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Sgemm(self):
        self._Tgemm('Sgemm', np.float32)

    def test_Dgemm(self):
        self._Tgemm('Dgemm', np.float64)

    def test_Cgemm(self):
        self._Tgemm('Cgemm', np.complex64)


    def test_Zgemm(self):
        self._Tgemm('Zgemm', np.complex128)


    def _Tsyrk(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)

        alpha = 1.2
        beta = .34

        uplo = 'U'
        trans = 'N'

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        getattr(blas, fn)(uplo, trans, n, k, alpha, dA, lda, beta, dC, ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Ssyrk(self):
        self._Tsyrk('Ssyrk', np.float32)

    def test_Dsyrk(self):
        self._Tsyrk('Dsyrk', np.float64)

    def test_Csyrk(self):
        self._Tsyrk('Csyrk', np.complex64)

    def test_Zsyrk(self):
        self._Tsyrk('Zsyrk', np.complex128)

    _Therk = _Tsyrk

    def test_Cherk(self):
        self._Therk('Cherk', np.complex64)

    def test_Zherk(self):
        self._Therk('Zherk', np.complex128)

    def _Tsymm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)

        alpha = 1.2
        beta = .34

        side = 'L'
        uplo =  'U'
        trans = 'N'

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        getattr(blas, fn)(side, uplo, m, n, alpha, dA, lda, dB, ldb, beta, dC,
                          ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Ssymm(self):
        self._Tsymm('Ssymm', np.float32)

    def test_Dsymm(self):
        self._Tsymm('Dsymm', np.float64)

    def test_Csymm(self):
        self._Tsymm('Csymm', np.complex64)
    
    def test_Zsymm(self):
        self._Tsymm('Zsymm', np.complex128)

    _Themm = _Tsymm

    def test_Chemm(self):
        self._Themm('Chemm', np.complex64)
    
    def test_Zhemm(self):
        self._Themm('Zhemm', np.complex128)

    def _Ttrsm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)
        dA = cuda.to_device(A)
        dB = cuda.to_device(B)

        alpha = 1.2
        beta = .34

        side = 'L'
        uplo =  'U'
        trans = 'N'
        diag = False

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        B0 = B.copy()
        getattr(blas, fn)(side, uplo, trans, diag, m, n, alpha, dA, lda, dB,
                          ldb)
        dB.copy_to_host(B)
        
        self.assertFalse(np.all(B == B0))

    def test_Strsm(self):
        self._Ttrsm('Strsm', np.float32)
    
    def test_Dtrsm(self):
        self._Ttrsm('Dtrsm', np.float64)

    def test_Ctrsm(self):
        self._Ttrsm('Ctrsm', np.complex64)
    
    def test_Ztrsm(self):
        self._Ttrsm('Ztrsm', np.complex128)

    def _Ttrmm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)

        alpha = 1.2
        beta = .34

        side = 'L'
        uplo =  'U'
        trans = 'N'
        diag = False

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        getattr(blas, fn)(side, uplo, trans, diag, m, n, alpha, dA, lda, dB,
                          ldb, dC, ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Strmm(self):
        self._Ttrmm('Strmm', np.float32)

    def test_Dtrmm(self):
        self._Ttrmm('Dtrmm', np.float64)

    def test_Ctrmm(self):
        self._Ttrmm('Ctrmm', np.complex64)
    
    def test_Ztrmm(self):
        self._Ttrmm('Ztrmm', np.complex128)


    def _Tdgmm(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 2.4], dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dx = cuda.to_device(x)
        dC = cuda.to_device(C)

        side = 'L'
        diag = False

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        incx = 1
        getattr(blas, fn)(side, m, n, dA, lda, dx, incx, dC, ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Sdgmm(self):
        self._Tdgmm('Sdgmm', np.float32)

    def test_Ddgmm(self):
        self._Tdgmm('Ddgmm', np.float64)

    def test_Cdgmm(self):
        self._Tdgmm('Cdgmm', np.complex64)
    
    def test_Zdgmm(self):
        self._Tdgmm('Zdgmm', np.complex128)


    def _Tgeam(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)
        dC = cuda.to_device(C)

        alpha = 1.2
        beta = .34

        transa = 'N'
        transb = 'N'

        blas = cuBlas()

        lda = ldb = ldc = m = n = k = 3
        C0 = C.copy()
        getattr(blas, fn)(transa, transb, m, n, alpha, dA, lda, beta, dB,
                          ldb, dC, ldc)
        dC.copy_to_host(C)
        
        self.assertFalse(np.all(C == C0))

    def test_Sgeam(self):
        self._Tgeam('Sgeam', np.float32)

    def test_Dgeam(self):
        self._Tgeam('Dgeam', np.float64)

    def test_Cgeam(self):
        self._Tgeam('Cgeam', np.complex64)
    
    def test_Zgeam(self):
        self._Tgeam('Zgeam', np.complex128)




class TestCuBlasAPI(unittest.TestCase):
    def setUp(self):
        from numbapro.cudalib.cublas import Blas
        self.blas = Blas()
        

    def Tnrm2(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        got = fn(x)
        exp = np.linalg.norm(x)
        self.assertTrue(np.allclose(got, exp))

    def test_nrm2(self):
        self.Tnrm2(self.blas.nrm2, np.float32)
        self.Tnrm2(self.blas.nrm2, np.float64)
        self.Tnrm2(self.blas.nrm2, np.complex64)
        self.Tnrm2(self.blas.nrm2, np.complex128)

    def Tdot(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        got = self.blas.dot(x, y)
        exp = np.dot(x, y)
        self.assertTrue(np.allclose(got, exp))

    def test_dot(self):
        self.Tdot(self.blas.dot, np.float32)
        self.Tdot(self.blas.dot, np.float64)
        self.Tdot(self.blas.dotu, np.complex64)
        self.Tdot(self.blas.dotu, np.complex128)

    def Tdotc(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        got = self.blas.dotc(x, y)
        exp = np.vdot(x, y)
        self.assertTrue(np.allclose(got, exp))

    def test_dotc(self):
        self.Tdot(self.blas.dotc, np.float32)

    def Tscal(self, fn, dtype, alpha):
        x = np.random.random(10).astype(dtype)
        x0 = x.copy()
        fn(alpha, x)
        self.assertTrue(np.allclose(x0 * alpha, x))

    def test_scal(self):
        self.Tscal(self.blas.scal, np.float32, 1.234)
        self.Tscal(self.blas.scal, np.float64, 1.234)
        self.Tscal(self.blas.scal, np.complex64, 1.234+5j)
        self.Tscal(self.blas.scal, np.complex128, 1.234+5j)
        self.Tscal(self.blas.scal, np.complex64, 1.234)
        self.Tscal(self.blas.scal, np.complex128, 1.234)

    def Taxpy(self, fn, dtype, alpha):
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        y0 = y.copy()

        fn(alpha, x, y)

        self.assertTrue(np.allclose(alpha * x + y0, y))

    def test_axpy(self):
        self.Taxpy(self.blas.axpy, np.float32, 1.234)
        self.Taxpy(self.blas.axpy, np.float64, 1.234)
        self.Taxpy(self.blas.axpy, np.complex64, 1.234j)
        self.Taxpy(self.blas.axpy, np.complex128, 1.234j)

    def Itamax(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        got = fn(x)
        self.assertTrue(np.allclose(np.argmax(x), got))

    def test_amax(self):
        self.Itamax(self.blas.amax, np.float32)
        self.Itamax(self.blas.amax, np.float64)
        self.Itamax(self.blas.amax, np.complex64)
        self.Itamax(self.blas.amax, np.complex128)

    def Itamin(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        got = fn(x)
        self.assertTrue(np.allclose(np.argmin(x), got))

    def test_amin(self):
        self.Itamin(self.blas.amin, np.float32)
        self.Itamin(self.blas.amin, np.float64)
        self.Itamin(self.blas.amin, np.complex64)
        self.Itamin(self.blas.amin, np.complex128)

    def Tasum(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        got = fn(x)
        self.assertTrue(np.allclose(np.sum(x), got))

    def test_asum(self):
        self.Tasum(self.blas.asum, np.float32)
        self.Tasum(self.blas.asum, np.float64)
        self.Tasum(self.blas.asum, np.complex64)
        self.Tasum(self.blas.asum, np.complex128)
    
    def Trot(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)
        angle = 1.342
        c = np.cos(angle)
        s = np.sin(angle)

        x0, y0 = c * x + s * y, -s * x + c * y

        fn(x, y, c, s)

        self.assertTrue(np.allclose(x, x0))
        self.assertTrue(np.allclose(y, y0))

    def test_rot(self):
        self.Trot(self.blas.rot, np.float32)
        self.Trot(self.blas.rot, np.float64)
        self.Trot(self.blas.rot, np.complex64)
        self.Trot(self.blas.rot, np.complex128)

    def Trotg(self, fn, dt1, dt2):
        a, b = (np.array(np.random.random(), dtype=dt1),
                np.array(np.random.random(), dtype=dt2))
        r, z, c, s = fn(a, b)

        rot = np.array([[c,           s],
                        [-np.conj(s), c]])
        vec = np.array([[a],
                        [b]])
        exp = np.dot(rot, vec)
        got = np.array([[r],
                        [0.0]])
        self.assertTrue(np.allclose(exp, got, atol=1e-6))

    def test_rotg(self):
        self.Trotg(self.blas.rotg, np.float32, np.float32)
        self.Trotg(self.blas.rotg, np.float64, np.float64)
        self.Trotg(self.blas.rotg, np.complex64, np.complex64)
        self.Trotg(self.blas.rotg, np.complex128, np.complex128)

    def Trotm(self, fn, dtype):
        x = np.random.random(10).astype(dtype)
        y = np.random.random(10).astype(dtype)

        param = np.random.random(5).astype(dtype)
        param[0] = -1.0
        h11, h21, h12, h22 = param[1:].tolist()

        x0, y0 = h11 * x + h12 * y, h21 * x + h22 * y

        fn(x, y, param)

        self.assertTrue(np.allclose(x, x0))
        self.assertTrue(np.allclose(y, y0))

    def test_rotm(self):
        self.Trotm(self.blas.rotm, np.float32)
        self.Trotm(self.blas.rotm, np.float64)

    def Trotmg(self, fn, dtype):
        d1, d2, x1, y1 = np.random.random(4).tolist()
        
        param = fn(d1, d2, x1, y1)

        flag, h11, h21, h12, h22 = param.tolist()

        if flag == -1.0:
            pass # don't know how to check
        elif flag == 0.0:
            self.assertEqual(h11, 0)
            self.assertEqual(h22, 0)
        elif flag == 1.0:
            self.assertEqual(h12, 0)
            self.assertEqual(h21, 0)
        else:
            self.assertEqual(flag, -2.0)
            self.assertEqual(h11, 0)
            self.assertEqual(h12, 0)
            self.assertEqual(h21, 0)
            self.assertEqual(h22, 0)

    def test_rotmg(self):
        self.Trotmg(self.blas.rotmg, np.float32)
        self.Trotmg(self.blas.rotmg, np.float64)

    # Level 2

    def _test_all(self, test, fn):
        dtypes = np.float32, np.float64, np.complex64, np.complex128
        for dt in dtypes:
            test(fn, dt)

    def _test_float(self, test, fn):
        dtypes = np.float32, np.float64
        for dt in dtypes:
            test(fn, dt)

    def _test_complex(self, test, fn):
        dtypes = np.complex64, np.complex128
        for dt in dtypes:
            test(fn, dt)
    
    def Tgbmv(self, fn, dtype):
        kl = 0
        ku = 0
        alpha = 1.
        beta = 0.
        A = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 3]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([1, 2, 3], dtype=dtype)
        lda, n = A.shape
        m = lda
        y0 = y.copy()
        fn('N', m, n, kl, ku, alpha, A, x, beta, y)
        self.assertFalse(all(y0 == y))

    def test_gbmv(self):
        self._test_all(self.Tgbmv, self.blas.gbmv)
    
    def Tgemv(self, fn, dtype):
        from numbapro.cudalib.cublas.binding import cuBlas
        blas = cuBlas()
        kl = 0
        ku = 0
        alpha = 1.
        beta = 0.
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([1, 2, 3], dtype=dtype)
        m, n = A.shape
        y0 = y.copy()

        fn('N', m, n, alpha, A, x, beta, y)
        self.assertFalse(all(y0 == y))

    def test_gemv(self):
        self._test_all(self.Tgemv, self.blas.gemv)

    def Ttrmv(self, fn, dtype):
        uplo = 'U'
        trans = 'N'
        diag = True
        n = 3
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        x0 = x.copy()
        fn(uplo, trans, diag, n, A, x)
        self.assertFalse(all(x == x0))

    def test_trmv(self):
        self._test_all(self.Ttrmv, self.blas.trmv)

    def Ttbmv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        k = 0
        fn(uplo, trans, diag, n, k, A, x)
        self.assertFalse(all(x == x0))

    def test_tbmv(self):
        self._test_all(self.Ttbmv, self.blas.tbmv)


    def Ttpmv(self, fn, dtype):
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)

        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        fn(uplo, trans, diag, n, AP, x)
        self.assertFalse(all(x == x0))

    def test_tpmv(self):
        self._test_all(self.Ttpmv, self.blas.tpmv)


    def Ttrsv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        fn(uplo, trans, diag, n, A, x)
        self.assertFalse(all(x == x0))

    def test_trsv(self):
        self._test_all(self.Ttrsv, self.blas.trsv)

    def Ttpsv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        x0 = x.copy()
        fn(uplo, trans, diag, n, A, x)
        self.assertFalse(all(x == x0))

    def test_tpsv(self):
        self._test_all(self.Ttpsv, self.blas.tpsv)

    def Ttbsv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        uplo = 'U'
        trans = 'N'
        diag = False
        n = 3
        k = 0
        x0 = x.copy()
        fn(uplo, trans, diag, n, k, A, x)
        self.assertFalse(all(x == x0))

    def test_tbsv(self):
        self._test_all(self.Ttbsv, self.blas.tbsv)


    def Tsymv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        beta = .34
        uplo = 'U'
        n = 3
        k = 0
        y0 = y.copy()
        fn(uplo, n, alpha, A, x, beta, y)
        self.assertFalse(all(y == y0))

    def test_symv(self):
        self._test_all(self.Tsymv, self.blas.symv)

    Themv = Tsymv

    def test_hemv(self):
        self._test_complex(self.Themv, self.blas.hemv)


    def Tsbmv(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        beta = .34
        uplo = 'U'
        n = 3
        k = 0
        y0 = y.copy()
        fn(uplo, n, k, alpha, A, x, beta, y)
        self.assertFalse(all(y == y0))

    def test_sbmv(self):
        self._test_float(self.Tsbmv, self.blas.sbmv)

    Thbmv = Tsbmv

    def test_hbmv(self):
        self._test_complex(self.Thbmv, self.blas.hbmv)

    def Tspmv(self, fn, dtype):
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        beta = .34
        uplo = 'U'
        n = 3
        k = 0
        y0 = y.copy()
        fn(uplo, n, alpha, AP, x, beta, y)
        self.assertFalse(all(y == y0))

    def test_spmv(self):
        self._test_float(self.Tspmv, self.blas.spmv)

    Thpmv = Tspmv

    def test_hpmv(self):
        self._test_complex(self.Thpmv, self.blas.hpmv)

    def Tger(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        m = n = 3
        A0 = A.copy()
        fn(m, n, alpha, x, y, A)
        self.assertFalse(np.all(A == A0))

    def test_ger(self):
        self._test_float(self.Tger, self.blas.ger)

    def test_geru(self):
        self._test_complex(self.Tger, self.blas.geru)

    def test_gerc(self):
        self._test_complex(self.Tger, self.blas.gerc)

    def Tsyr(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)

        alpha = 1.2
        uplo = 'U'
        m = n = 3
        A0 = A.copy()
        fn(uplo, n, alpha, x, A)
        self.assertFalse(np.all(A == A0))

    def test_syr(self):
        self._test_all(self.Tsyr, self.blas.syr)

    def Ther(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        alpha = 1.2
        uplo = 'U'
        m = n = 3
        A0 = A.copy()
        fn(uplo, n, alpha, x, A)
        self.assertFalse(np.all(A == A0))

    def test_her(self):
        self._test_complex(self.Ther, self.blas.her)

    def Tspr(self, fn, dtype):
        AP = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        alpha = 1.2
        uplo = 'U'
        m = n = 3
        AP0 = AP.copy()
        fn(uplo, n, alpha, x, AP)
        self.assertFalse(np.all(AP == AP0))

    def test_spr(self):
        self._test_float(self.Tspr, self.blas.spr)

    Thpr = Tspr

    def test_hpr(self):
        self._test_complex(self.Thpr, self.blas.hpr)

    def Tsyr2(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        uplo = 'U'
        m = n = 3
        A0 = A.copy()
        fn(uplo, n, alpha, x, y, A)
        self.assertFalse(np.all(A == A0))

    Ther2 = Tsyr2

    def test_syr2(self):
        self._test_all(self.Tsyr2, self.blas.syr2)

    def test_her2(self):
        self._test_complex(self.Ther2, self.blas.her2)

    def Tspr2(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([8, 2, 3], dtype=dtype)
        alpha = 1.2
        uplo = 'U'
        n = 3
        A0 = A.copy()
        fn(uplo, n, alpha, x, y, A)
        self.assertFalse(np.all(A == A0))

    Thpr2 = Tspr2

    def test_spr2(self):
        self._test_float(self.Tspr2, self.blas.spr2)

    def test_hpr2(self):
        self._test_complex(self.Thpr2, self.blas.hpr2)

    # Level 3

    def Tgemm(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        transa = 'N'
        transb = 'N'
        m = n = k = 3
        C0 = C.copy()
        fn(transa, transb, m, n, k, alpha, A, B, beta, C)
        self.assertFalse(np.all(C == C0))

    def test_gemm(self):
        self._test_all(self.Tgemm, self.blas.gemm)


    def Tsyrk(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        uplo = 'U'
        trans = 'N'
        m = n = k = 3
        C0 = C.copy()
        fn(uplo, trans, n, k, alpha, A, beta, C)
        self.assertFalse(np.all(C == C0))

    def test_syrk(self):
        self._test_all(self.Tsyrk, self.blas.syrk)

    Therk = Tsyrk

    def test_herk(self):
        self._test_complex(self.Therk, self.blas.herk)

    def Tsymm(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        side = 'L'
        uplo =  'U'
        trans = 'N'
        m = n = k = 3
        C0 = C.copy()
        fn(side, uplo, m, n, alpha, A, B, beta, C)
        self.assertFalse(np.all(C == C0))

    def test_symm(self):
        self._test_all(self.Tsymm, self.blas.symm)

    Themm = Tsymm

    def test_hemm(self):
        self._test_complex(self.Themm, self.blas.hemm)

    def Ttrsm(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        side = 'L'
        uplo =  'U'
        trans = 'N'
        diag = False
        m = n = k = 3
        B0 = B.copy()
        fn(side, uplo, trans, diag, m, n, alpha, A, B)
        self.assertFalse(np.all(B == B0))

    def test_trsm(self):
        self._test_all(self.Ttrsm, self.blas.trsm)

    def Ttrmm(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        side = 'L'
        uplo =  'U'
        trans = 'N'
        diag = False
        m = n = k = 3
        C0 = C.copy()
        fn(side, uplo, trans, diag, m, n, alpha, A, B, C)
        self.assertFalse(np.all(C == C0))

    def test_trmm(self):
        self._test_all(self.Ttrmm, self.blas.trmm)

    def Tdgmm(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        x = np.array([1, 2, 2.4], dtype=dtype)
        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        side = 'L'
        diag = False
        m = n = k = 3
        C0 = C.copy()
        fn(side, m, n, A, x, C)
        self.assertFalse(np.all(C == C0))

    def test_dgmm(self):
        self._test_all(self.Tdgmm, self.blas.dgmm)


    def Tgeam(self, fn, dtype):
        A = np.array([[1, 2, 0],
                      [0, 3, 0],
                      [1, 0, 1]], order='F', dtype=dtype)
        B = np.array([[2, 2, 0],
                      [7, 0, 0],
                      [1, 4, 1]], order='F', dtype=dtype)

        C = np.array([[0, 9, 0],
                      [0, 1, 1],
                      [0, 0, 1]], order='F', dtype=dtype)
        alpha = 1.2
        beta = .34
        transa = 'N'
        transb = 'N'
        m = n = k = 3
        C0 = C.copy()
        fn(transa, transb, m, n, alpha, A, beta, B, C)
        self.assertFalse(np.all(C == C0))

    def test_geam(self):
        self._test_all(self.Tgeam, self.blas.geam)

if __name__ == '__main__':
    unittest.main()

