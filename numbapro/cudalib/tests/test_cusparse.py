import numpy as np
import unittest
from .support import addtest, main
from numbapro.cudalib.cusparse import Sparse
from numbapro import cuda


@addtest
class TestCuSparseAPI(unittest.TestCase):
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

if __name__ == '__main__':
    main()

