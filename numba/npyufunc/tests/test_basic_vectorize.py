import numpy as np
from numba import int32, uint32, float32, float64, complex64
from numba.vectorize import Vectorize
from timeit import default_timer as time
import unittest

def vector_add(a, b):
    return a + b

class TestBasicVectorize(unittest.TestCase):
    def setUp(self):
        bv = Vectorize(vector_add, backend='ast')
        bv.add(restype=int32,   argtypes=[int32,    int32])
        bv.add(restype=uint32,  argtypes=[uint32,   uint32])
        bv.add(restype=float32, argtypes=[float32,  float32])
        bv.add(restype=float64, argtypes=[float64, 	float64])
        bv.add(restype=complex64, argtypes=[complex64, complex64])
        self.basic_ufunc = bv.build_ufunc()

    def _test(self, ty):
        print("Test %s" % ty)
        data = np.linspace(0., 10000., 100000).astype(ty)

        ts = time()
        result = self.basic_ufunc(data, data)
        tnumba = time() - ts

        ts = time()
        gold = np.add(data, data)
        tnumpy = time() - ts

        print("Numpy time: %fs" % tnumpy)
        print("Numba time: %fs" % tnumba)

        if tnumba < tnumpy:
            print("Numba is FASTER by %fx" % (tnumpy/tnumba))
        else:
            print("Numba is SLOWER by %fx" % (tnumba/tnumpy))


        for expect, got in zip(gold, result):
            assert expect == got

    def test_double(self):
        self._test(np.double)

    def test_float(self):
        self._test(np.float)

    def test_int32(self):
        self._test(np.int32)

    def test_uint32(self):
        self._test(np.uint32)

    def test_complex64(self):
        self._test(np.complex64)

if __name__ == '__main__':
    unittest.main()
