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
        x = np.arange(10, dtype=dtype)
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


if __name__ == '__main__':
    unittest.main()
