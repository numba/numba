import numpy as np
import unittest
from .support import addtest, main
from numbapro.cudalib.cusparse import Sparse
from numbapro import cuda


@addtest
class TestCuSparseLevel1(unittest.TestCase):
    def setUp(self):
        self.cus = Sparse()

    def generic_test_axpyi(self, dtype):
        alpha = 2
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.zeros(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.axpyi(alpha, xval, xind, y)
        self.assertTrue(np.allclose(y[xind], (xval * 2)))

    def test_Saxpyi(self):
        self.generic_test_axpyi(dtype=np.float32)

    def test_Daxpyi(self):
        self.generic_test_axpyi(dtype=np.float64)

    def test_Caxpyi(self):
        self.generic_test_axpyi(dtype=np.complex64)

    def test_Zaxpyi(self):
        self.generic_test_axpyi(dtype=np.complex128)

    def generic_test_doti(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        result = self.cus.doti(xval, xind, y)
        self.assertTrue(result)

    def test_Sdoti(self):
        self.generic_test_doti(dtype=np.float32)

    def test_Zdoti(self):
        self.generic_test_doti(dtype=np.complex128)

    def generic_test_dotci(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        result = self.cus.dotci(xval, xind, y)
        self.assertTrue(result)

    def test_Zdotci(self):
        self.generic_test_dotci(dtype=np.complex128)

    def generic_test_gthr(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.gthr(y, xval, xind)
        self.assertTrue(np.all(xval == 1))

    def test_Sgthr(self):
        self.generic_test_gthr(dtype=np.float32)

    def test_Cgthr(self):
        self.generic_test_gthr(dtype=np.complex64)

    def generic_test_gthrz(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.gthrz(y, xval, xind)
        self.assertTrue(np.all(xval == 1))
        self.assertTrue(np.all(y[xind] == 0))

    def test_Dgthr(self):
        self.generic_test_gthrz(dtype=np.float64)

    def test_Zgthr(self):
        self.generic_test_gthrz(dtype=np.complex128)

    def generic_test_roti(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        c = .2
        s = .3
        oldxval = xval.copy()
        oldy = y.copy()
        self.cus.roti(xval, xind, y, c, s)
        self.assertFalse(np.all(oldxval == xval))
        self.assertFalse(np.all(oldy == y))

    def test_Sroti(self):
        self.generic_test_roti(dtype=np.float32)

    def test_Droti(self):
        self.generic_test_roti(dtype=np.float64)

    def generic_test_sctr(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        oldy = y.copy()
        self.cus.sctr(xval, xind, y)
        self.assertFalse(np.all(oldy == y))

    def test_Ssctr(self):
        self.generic_test_sctr(dtype=np.float32)

    def test_Csctr(self):
        self.generic_test_sctr(dtype=np.complex64)


if __name__ == '__main__':
    main()

