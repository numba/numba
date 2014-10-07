from __future__ import print_function, absolute_import, division

import itertools

import numpy as np

from numba import utils
from numba import unittest_support as unittest
from .support import TestCase


DBL_EPSILON = 2**-52
FLT_EPSILON = 2**-23

INF = float('inf')
NAN = float('nan')


class TestAssertPreciseEqual(TestCase):
    """
    Tests for TestCase.assertPreciseEqual().
    """

    def eq(self, left, right):
        def assert_succeed(left, right):
            self.assertPreciseEqual(left, right)
            self.assertPreciseEqual(right, left)
        assert_succeed(left, right)
        assert_succeed((left, left), (right, right))

    def ne(self, left, right):
        def assert_fail(left, right):
            try:
                self.assertPreciseEqual(left, right)
            except AssertionError:
                pass
            else:
                self.fail("%s and %s unexpectedly considered equal" % (left, right))
        assert_fail(left, right)
        assert_fail(right, left)
        assert_fail((left, left), (right, right))
        assert_fail((right, right), (left, left))

    def test_types(self):
        # assertPreciseEqual() should test for type compatibility
        int_types = [int]
        if utils.PYVERSION < (3,):
            int_types.append(long)
        np_float_types = [np.float32, np.float64]
        float_types = [float] + np_float_types
        np_complex_types = [np.complex64, np.complex128]
        complex_types = [complex]
        # int-like, float-like, complex-like are not compatible
        for i, f, c in itertools.product(int_types, float_types, complex_types):
            self.ne(i(1), f(1))
            self.ne(f(1), c(1))
            self.ne(i(1), c(1))
        # int and long are compatible between each other
        for u, v in itertools.product(int_types, int_types):
            self.eq(u(1), v(1))
        # NumPy float types are not compatible between each other
        for u, v in itertools.product(np_float_types, np_float_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))
        # NumPy complex types are not compatible between each other
        for u, v in itertools.product(np_complex_types, np_complex_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))

    def test_int_values(self):
        self.eq(0, 0)
        self.ne(0, 1)
        self.ne(-1, 1)

    def test_float_values(self):
        for tp in [float, np.float32, np.float64]:
            self.eq(tp(1.5), tp(1.5))
            # Infinities
            self.eq(tp(INF), tp(INF))
            self.ne(tp(INF), tp(1e38))
            self.eq(tp(-INF), tp(-INF))
            self.ne(tp(INF), tp(-INF))
            # NaNs
            self.eq(tp(NAN), tp(NAN))
            self.ne(tp(NAN), tp(0))
            self.ne(tp(NAN), tp(INF))
            self.ne(tp(NAN), tp(-INF))
            # Signed zeros
            self.eq(tp(0.0), tp(0.0))
            self.eq(tp(-0.0), tp(-0.0))
            # FIXME
            #self.ne(tp(0.0), tp(-0.0))

    def test_float64_values(self):
        for tp in [float, np.float64]:
            self.ne(tp(1.0 + DBL_EPSILON), tp(1.0))

    def test_float32_values(self):
        tp = np.float32
        self.ne(tp(1.0 + FLT_EPSILON), tp(1.0))

    def test_complex_values(self):
        for tp in [complex, np.complex64, np.complex128]:
            self.eq(tp(1 + 2j), tp(1 + 2j))
            self.ne(tp(1 + 1j), tp(1 + 2j))
            self.ne(tp(2 + 2j), tp(1 + 2j))
            # Infinities
            self.eq(tp(complex(INF, INF)), tp(complex(INF, INF)))
            self.eq(tp(complex(INF, -INF)), tp(complex(INF, -INF)))
            self.eq(tp(complex(-INF, -INF)), tp(complex(-INF, -INF)))
            self.ne(tp(complex(INF, INF)), tp(complex(INF, -INF)))
            self.ne(tp(complex(INF, INF)), tp(complex(-INF, INF)))
            self.eq(tp(complex(INF, 0)), tp(complex(INF, 0)))
            self.ne(tp(complex(INF, 0)), tp(complex(INF, 1)))
            # NaNs
            self.eq(tp(complex(NAN, 0)), tp(complex(NAN, 0)))
            self.eq(tp(complex(0, NAN)), tp(complex(0, NAN)))
            self.eq(tp(complex(NAN, NAN)), tp(complex(NAN, NAN)))
            self.eq(tp(complex(INF, NAN)), tp(complex(INF, NAN)))
            self.eq(tp(complex(NAN, -INF)), tp(complex(NAN, -INF)))
            # FIXME
            #self.ne(tp(complex(NAN, INF)), tp(complex(NAN, -INF)))
            #self.ne(tp(complex(NAN, 0)), tp(complex(NAN, 1)))
            #self.ne(tp(complex(INF, NAN)), tp(complex(-INF, NAN)))
            #self.ne(tp(complex(0, NAN)), tp(complex(1, NAN)))
            #self.ne(tp(complex(NAN, 0)), tp(complex(0, NAN)))


if __name__ == '__main__':
    unittest.main()

