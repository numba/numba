from __future__ import print_function
import numba.unittest_support as unittest
from numba import types, utils, jit
from numba.tests import usecases
import numpy as np

@jit(nopython=True)
def add_5(x):
    x += 5


class TestInplaceAdd(unittest.TestCase):

    def test_int(self):
        arr = np.arange(10)
        add_5(arr)
        res = np.arange(10) + 5
        for (i, j) in zip(arr, res):
            self.assertEqual(i, j)


if __name__ == '__main__':
    unittest.main()
