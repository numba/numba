from __future__ import print_function, division, absolute_import

import numpy as np

from numba import njit
from numba import unittest_support as unittest


class TestParfors(unittest.TestCase):

    def test_arraymap(self):
        @njit(parallel=True)
        def axy(a, x, y):
            return a * x + y

        A = np.linspace(0,1,10)
        X = np.linspace(2,1,10)
        Y = np.linspace(1,2,10)

        output = axy(A,X,Y)
        expected = A*X+Y
        np.testing.assert_array_equal(expected, output)
        self.assertIn('@do_scheduling', axy.inspect_llvm(axy.signatures[0]))

    def test_mvdot(self):
        @njit(parallel=True)
        def ddot(a, v):
            return np.dot(a,v)

        A = np.linspace(0,1,20).reshape(2,10)
        v = np.linspace(2,1,10)

        output = ddot(A,v)
        expected = np.dot(A,v)
        np.testing.assert_array_almost_equal(expected, output, decimal=5)
        self.assertIn('@do_scheduling', ddot.inspect_llvm(ddot.signatures[0]))

if __name__ == "__main__":
    unittest.main()
