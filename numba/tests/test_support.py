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

    int_types = [int]
    if utils.PYVERSION < (3,):
        int_types.append(long)
    np_float_types = [np.float32, np.float64]
    float_types = [float] + np_float_types
    np_complex_types = [np.complex64, np.complex128]
    complex_types = [complex] + np_complex_types

    def eq(self, left, right, **kwargs):
        def assert_succeed(left, right):
            self.assertPreciseEqual(left, right, **kwargs)
            self.assertPreciseEqual(right, left, **kwargs)
        assert_succeed(left, right)
        assert_succeed((left, left), (right, right))
        assert_succeed([left, left], [right, right])

    def ne(self, left, right, **kwargs):
        def assert_fail(left, right):
            try:
                self.assertPreciseEqual(left, right, **kwargs)
            except AssertionError:
                pass
            else:
                self.fail("%s and %s unexpectedly considered equal" % (left, right))
        assert_fail(left, right)
        assert_fail(right, left)
        assert_fail((left, left), (right, right))
        assert_fail((right, right), (left, left))
        assert_fail([left, left], [right, right])
        assert_fail([right, right], [left, left])

    def test_types(self):
        # assertPreciseEqual() should test for type compatibility
        # int-like, float-like, complex-like are not compatible
        for i, f, c in itertools.product(self.int_types, self.float_types,
                                         self.complex_types):
            self.ne(i(1), f(1))
            self.ne(f(1), c(1))
            self.ne(i(1), c(1))
        # int and long are compatible between each other
        for u, v in itertools.product(self.int_types, self.int_types):
            self.eq(u(1), v(1))
        # NumPy float types are not compatible between each other
        for u, v in itertools.product(self.np_float_types, self.np_float_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))
        # NumPy complex types are not compatible between each other
        for u, v in itertools.product(self.np_complex_types, self.np_complex_types):
            if u is v:
                self.eq(u(1), v(1))
            else:
                self.ne(u(1), v(1))

    def test_int_values(self):
        for tp in self.int_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(0), tp(0), prec=prec)
                self.ne(tp(0), tp(1), prec=prec)
                self.ne(tp(-1), tp(1), prec=prec)
                self.ne(tp(2**80), tp(1+2**80), prec=prec)

    def test_float_values(self):
        for tp in self.float_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(1.5), tp(1.5), prec=prec)
                # Signed zeros
                self.eq(tp(0.0), tp(0.0), prec=prec)
                self.eq(tp(-0.0), tp(-0.0), prec=prec)
                self.ne(tp(0.0), tp(-0.0), prec=prec)
                # Infinities
                self.eq(tp(INF), tp(INF), prec=prec)
                self.ne(tp(INF), tp(1e38), prec=prec)
                self.eq(tp(-INF), tp(-INF), prec=prec)
                self.ne(tp(INF), tp(-INF), prec=prec)
                # NaNs
                self.eq(tp(NAN), tp(NAN), prec=prec)
                self.ne(tp(NAN), tp(0), prec=prec)
                self.ne(tp(NAN), tp(INF), prec=prec)
                self.ne(tp(NAN), tp(-INF), prec=prec)

    def test_float64_values(self):
        for tp in [float, np.float64]:
            self.ne(tp(1.0 + DBL_EPSILON), tp(1.0))

    def test_float32_values(self):
        tp = np.float32
        self.ne(tp(1.0 + FLT_EPSILON), tp(1.0))

    def test_float64_values_inexact(self):
        for tp in [float, np.float64]:
            for scale in [1.0, -2**3, 2**-4, -2**-20]:
                a = scale * 1.0
                b = scale * (1.0 + DBL_EPSILON)
                c = scale * (1.0 + DBL_EPSILON * 2)
                d = scale * (1.0 + DBL_EPSILON * 4)
                self.ne(tp(a), tp(b))
                self.ne(tp(a), tp(b), prec='exact')
                self.eq(tp(a), tp(b), prec='double')
                self.eq(tp(a), tp(b), prec='double', ulps=1)
                self.ne(tp(a), tp(c), prec='double')
                self.eq(tp(a), tp(c), prec='double', ulps=2)
                self.ne(tp(a), tp(d), prec='double', ulps=2)
                self.eq(tp(a), tp(c), prec='double', ulps=3)
                self.eq(tp(a), tp(d), prec='double', ulps=3)

    def test_float32_values_inexact(self):
        tp = np.float32
        for scale in [1.0, -2**3, 2**-4, -2**-20]:
            # About the choice of 0.9: there seem to be issues when
            # converting
            a = scale * 1.0
            b = scale * (1.0 + FLT_EPSILON)
            c = scale * (1.0 + FLT_EPSILON * 2)
            d = scale * (1.0 + FLT_EPSILON * 4)
            self.ne(tp(a), tp(b))
            self.ne(tp(a), tp(b), prec='exact')
            self.ne(tp(a), tp(b), prec='double')
            self.eq(tp(a), tp(b), prec='single')
            self.ne(tp(a), tp(c), prec='single')
            self.eq(tp(a), tp(c), prec='single', ulps=2)
            self.ne(tp(a), tp(d), prec='single', ulps=2)
            self.eq(tp(a), tp(c), prec='single', ulps=3)
            self.eq(tp(a), tp(d), prec='single', ulps=3)

    def test_complex_values(self):
        # Complex literals with signed zeros are confusing, better use
        # the explicit constructor.
        c_pp, c_pn, c_np, c_nn = [complex(0.0, 0.0), complex(0.0, -0.0),
                                  complex(-0.0, 0.0), complex(-0.0, -0.0)]
        for tp in self.complex_types:
            for prec in ['exact', 'single', 'double']:
                self.eq(tp(1 + 2j), tp(1 + 2j), prec=prec)
                self.ne(tp(1 + 1j), tp(1 + 2j), prec=prec)
                self.ne(tp(2 + 2j), tp(1 + 2j), prec=prec)
                # Signed zeros
                self.eq(tp(c_pp), tp(c_pp), prec=prec)
                self.eq(tp(c_np), tp(c_np), prec=prec)
                self.eq(tp(c_nn), tp(c_nn), prec=prec)
                self.ne(tp(c_pp), tp(c_pn), prec=prec)
                self.ne(tp(c_pn), tp(c_nn), prec=prec)
                # Infinities
                self.eq(tp(complex(INF, INF)), tp(complex(INF, INF)), prec=prec)
                self.eq(tp(complex(INF, -INF)), tp(complex(INF, -INF)), prec=prec)
                self.eq(tp(complex(-INF, -INF)), tp(complex(-INF, -INF)), prec=prec)
                self.ne(tp(complex(INF, INF)), tp(complex(INF, -INF)), prec=prec)
                self.ne(tp(complex(INF, INF)), tp(complex(-INF, INF)), prec=prec)
                self.eq(tp(complex(INF, 0)), tp(complex(INF, 0)), prec=prec)
                # NaNs
                self.eq(tp(complex(NAN, 0)), tp(complex(NAN, 0)), prec=prec)
                self.eq(tp(complex(0, NAN)), tp(complex(0, NAN)), prec=prec)
                self.eq(tp(complex(NAN, NAN)), tp(complex(NAN, NAN)), prec=prec)
                self.eq(tp(complex(INF, NAN)), tp(complex(INF, NAN)), prec=prec)
                self.eq(tp(complex(NAN, -INF)), tp(complex(NAN, -INF)), prec=prec)
                # FIXME
                #self.ne(tp(complex(NAN, INF)), tp(complex(NAN, -INF)))
                #self.ne(tp(complex(NAN, 0)), tp(complex(NAN, 1)))
                #self.ne(tp(complex(INF, NAN)), tp(complex(-INF, NAN)))
                #self.ne(tp(complex(0, NAN)), tp(complex(1, NAN)))
                #self.ne(tp(complex(NAN, 0)), tp(complex(0, NAN)))
            # XXX should work with other precisions as well?
            self.ne(tp(complex(INF, 0)), tp(complex(INF, 1)), prec='exact')

    def test_complex128_values_inexact(self):
        for tp in [complex, np.complex128]:
            for scale in [1.0, -2**3, 2**-4, -2**-20]:
                a = scale * 1.0
                b = scale * (1.0 + DBL_EPSILON)
                c = scale * (1.0 + DBL_EPSILON * 2)
                aa = tp(complex(a, a))
                ab = tp(complex(a, b))
                bb = tp(complex(b, b))
                self.ne(tp(aa), tp(ab))
                self.eq(tp(aa), tp(ab), prec='double')
                self.eq(tp(ab), tp(bb), prec='double')
                self.eq(tp(aa), tp(bb), prec='double')
                ac = tp(complex(a, c))
                cc = tp(complex(c, c))
                self.ne(tp(aa), tp(ac), prec='double')
                self.ne(tp(ac), tp(cc), prec='double')
                self.eq(tp(aa), tp(ac), prec='double', ulps=2)
                self.eq(tp(ac), tp(cc), prec='double', ulps=2)
                self.eq(tp(aa), tp(cc), prec='double', ulps=2)
                self.eq(tp(aa), tp(cc), prec='single')

    def test_complex64_values_inexact(self):
        tp = np.complex64
        for scale in [1.0, -2**3, 2**-4, -2**-20]:
            a = scale * 1.0
            b = scale * (1.0 + FLT_EPSILON)
            c = scale * (1.0 + FLT_EPSILON * 2)
            aa = tp(complex(a, a))
            ab = tp(complex(a, b))
            bb = tp(complex(b, b))
            self.ne(tp(aa), tp(ab))
            self.ne(tp(aa), tp(ab), prec='double')
            self.eq(tp(aa), tp(ab), prec='single')
            self.eq(tp(ab), tp(bb), prec='single')
            self.eq(tp(aa), tp(bb), prec='single')
            ac = tp(complex(a, c))
            cc = tp(complex(c, c))
            self.ne(tp(aa), tp(ac), prec='single')
            self.ne(tp(ac), tp(cc), prec='single')
            self.eq(tp(aa), tp(ac), prec='single', ulps=2)
            self.eq(tp(ac), tp(cc), prec='single', ulps=2)
            self.eq(tp(aa), tp(cc), prec='single', ulps=2)

    def test_arrays(self):
        a = np.arange(1, 7, dtype=np.int16).reshape((2, 3))
        b = a.copy()
        self.eq(a, b)
        # Different values
        self.ne(a, b + 1)
        self.ne(a, b[:-1])
        self.ne(a, b.T)
        # Different dtypes
        self.ne(a, b.astype(np.int32))
        # Different layout
        self.ne(a, b.T.copy().T)
        # Different ndim
        self.ne(a, b.flatten())
        # Precision
        a = np.arange(1, 3, dtype=np.float64)
        b = a * (1.0 + DBL_EPSILON)
        c = a * (1.0 + DBL_EPSILON * 2)
        self.ne(a, b)
        self.eq(a, b, prec='double')
        self.ne(a, c, prec='double')


if __name__ == '__main__':
    unittest.main()

