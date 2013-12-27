from __future__ import print_function
import unittest
import itertools
from numba.compiler import compile_isolated
from numba import types
from numba.tests import usecases


class TestUsecases(unittest.TestCase):
    def test_andor(self):
        pyfunc = usecases.andor
        ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))

        # Argument boundaries
        xs = -1, 0, 1, 9, 10, 11
        ys = -1, 0, 1, 9, 10, 11

        for args in itertools.product(xs, ys):
            print("case x=%d, y=%d" % args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_sum1d(self):
        pyfunc = usecases.sum1d
        ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case s=%d, e=%d" % args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_sum2d(self):
        pyfunc = usecases.sum2d
        ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case s=%d, e=%d" % args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_while_count(self):
        pyfunc = usecases.while_count
        ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case s=%d, e=%d" % args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

if __name__ == '__main__':
    unittest.main()

