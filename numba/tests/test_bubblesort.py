from __future__ import print_function

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def bubblesort(X):
    N = X.shape[0]
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp


class TestBubbleSort(unittest.TestCase):
    def test_bubblesort(self):
        pyfunc = bubblesort
        aryty = types.Array(dtype=types.int64, ndim=1, layout='C')
        cr = compile_isolated(pyfunc, (aryty,))
        cfunc = cr.entry_point

        array = np.array(list(reversed(range(8))), dtype="int64")
        control = array.copy()

        cfunc(array)
        pyfunc(control)

        self.assertTrue((array == control).all())


if __name__ == '__main__':
    unittest.main()
