import numpy as np
import unittest

from numba import void, int32, uint32, jit, int64

@jit(void(uint32[:], uint32, uint32))
def prng(X, A, C):
    for i in range(X.shape[0]):
        for j in range(100):
            v = (A * X[i] + C)
            X[i] = v & 0xffffffff

@jit(uint32())
def unsigned_literal():
    return abs(0xFFFFFFFF)

@jit(int64())
def unsigned_literal_64():
    return 0x100000000

@jit(int64(int32))
def constant_int_add(a):
    return 0xffffffff + a

class Test(unittest.TestCase):
    def test_prng(self):
        N = 100
        A = 1664525
        C = 1013904223
        X0 = np.arange(N, dtype=np.uint32)
        X1 = X0.copy()
        prng.py_func(X0, A, C)
        prng(X1, A, C)
        self.assertTrue(np.all(X1 >= 0))
        self.assertTrue(np.all(X0 == X1))

    def test_unsigned_literal(self):
        got = unsigned_literal()
        expect = abs(0xFFFFFFFF)
        self.assertEqual(expect, got)

    def test_unsigned_literal_64(self):
        got = unsigned_literal_64()
        expect = 0x100000000
        self.assertEqual(expect, got)

    def test_constant_int_add(self):
        got = constant_int_add(1)
        expect = 0xffffffff + 1
        self.assertEqual(expect, got)

if __name__ == '__main__':
    unittest.main()
