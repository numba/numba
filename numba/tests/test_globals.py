from __future__ import print_function, division, absolute_import
import numpy as np
from numba import jit
from numba import unittest_support as unittest


X = np.arange(10)


def global_ndarray_func(x):
    y = x + X.shape[0]
    return y


class TestGlobals(unittest.TestCase):
    def test_global_ndarray(self):
        # (see github issue #448)
        ctestfunc = jit('i4(i4)')(global_ndarray_func)
        self.assertEqual(ctestfunc(1), 11)


if __name__ == '__main__':
    unittest.main()
