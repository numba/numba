from __future__ import print_function
import unittest
import itertools
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types
from numba.tests import usecases

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")


class TestUsecases(unittest.TestCase):
    def test_andor(self):
        pyfunc = usecases.andor
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32))

        # Argument boundaries
        xs = -1, 0, 1, 9, 10, 11
        ys = -1, 0, 1, 9, 10, 11

        for args in itertools.product(xs, ys):
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_sum1d(self):
        pyfunc = usecases.sum1d
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_sum2d(self):
        pyfunc = usecases.sum2d
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_while_count(self):
        pyfunc = usecases.while_count
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32))

        ss = -1, 0, 1, 100, 200
        es = -1, 0, 1, 100, 200

        for args in itertools.product(ss, es):
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_copy_arrays(self):
        pyfunc = usecases.copy_arrays
        arraytype = types.Array(types.int32, 1, 'A')
        ctx, cfunc, err = compile_isolated(pyfunc, (arraytype, arraytype))

        nda = 0, 1, 10, 100

        for nd in nda:
            a = np.arange(nd, dtype='int32')
            b = np.empty_like(a)
            args = a, b

            print("case", args)
            cfunc(*args)
            self.assertTrue(np.all(a == b))

    def test_copy_arrays2d(self):
        pyfunc = usecases.copy_arrays2d
        arraytype = types.Array(types.int32, 2, 'A')
        ctx, cfunc, err = compile_isolated(pyfunc, (arraytype, arraytype))

        nda = (0, 0), (1, 1), (2, 5), (4, 25)

        for nd in nda:
            d1, d2 = nd
            a = np.arange(d1 * d2, dtype='int32').reshape(d1, d2)
            b = np.empty_like(a)
            args = a, b

            print("case", args)
            cfunc(*args)
            self.assertTrue(np.all(a == b))

    def test_ifelse1(self):
        pyfunc = usecases.ifelse1
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32))

        xs = -1, 0, 1
        ys = -1, 0, 1

        for x, y in itertools.product(xs, ys):
            args = x, y
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_string1(self):
        pyfunc = usecases.string1
        ctx, cfunc, err = compile_isolated(pyfunc, (types.int32, types.int32),
                                           flags=enable_pyobj_flags)

        xs = -1, 0, 1
        ys = -1, 0, 1

        for x, y in itertools.product(xs, ys):
            args = x, y
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_blackscholes_cnd(self):
        pyfunc = usecases.blackscholes_cnd
        ctx, cfunc, err = compile_isolated(pyfunc, (types.float32,))

        ds = -0.5, 0, 0.5

        for d in ds:
            args = (d,)
            print("case", args)
            self.assertEqual(pyfunc(*args), cfunc(*args))


if __name__ == '__main__':
    unittest.main()

