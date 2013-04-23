import numpy as np
import unittest

from numbapro import cuda

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

    def test_Cger(self):
        self._Tger('Cgeru', np.complex64)

    def test_Cger(self):
        self._Tger('Cgerc', np.complex64)

    def test_Zger(self):
        self._Tger('Zgeru', np.complex128)

    def test_Zger(self):
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

    def test_Ssyr(self):
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

    def test_Zspr2(self):
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

if __name__ == '__main__':
    unittest.main()

