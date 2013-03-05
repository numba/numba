import unittest, random
import numpy as np

from numbapro import cuda
from numba import jit, void, int32, int64

@cuda.jit(void(int32[:], int32, int32))
def cu_mod_int32(out, a, b):
    out[0] = a % b

@cuda.jit(void(int64[:], int64, int64))
def cu_mod_int64(out, a, b):
    out[0] = a % b

griddim  = 1,
blockdim = 1,

class TestMod(unittest.TestCase):
    def test_int32(self):
        for _ in xrange(100):
            a = random.randint(0, 2**31-1)
            b = random.randint(0, 2**31-1)
            self._test_template(cu_mod_int32, np.int32, a, b)

    def test_int64(self):
        for _ in xrange(100):
            a = random.randint(0, 2**63-1)
            b = random.randint(0, 2**63-1)
            self._test_template(cu_mod_int64, np.int64, a, b)

    def _test_template(self, cukernel, npydtype, a, b):
        out = np.empty(1, dtype=npydtype)
        expect = a % b
        cukernel[griddim, blockdim](out, a, b)
        got = out[0]
        self.assertEqual(got, expect)

if __name__ == '__main__':
    unittest.main()
