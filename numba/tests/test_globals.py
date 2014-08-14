from __future__ import print_function, division, absolute_import
import numpy as np
from numba import jit
from numba import unittest_support as unittest


X = np.arange(10)


def global_ndarray_func(x):
    y = x + X.shape[0]
    return y


class TestGlobals(unittest.TestCase):

    def check_global_ndarray(self, **jitargs):
        # (see github issue #448)
        ctestfunc = jit(**jitargs)(global_ndarray_func)
        self.assertEqual(ctestfunc(1), 11)

    def test_global_ndarray(self):
        # This also checks we can access an unhashable global value
        # (see issue #697)
        self.check_global_ndarray(forceobj=True)

    def test_global_ndarray_npm(self):
        self.check_global_ndarray(nopython=True)


if __name__ == '__main__':
    unittest.main()
