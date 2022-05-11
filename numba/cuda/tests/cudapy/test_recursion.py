import math
import warnings

from numba import cuda, jit
from numba.core.errors import TypingError, NumbaWarning
from numba.cuda.testing import CUDATestCase
import numpy as np
import unittest


class TestSelfRecursion(CUDATestCase):

    def setUp(self):
        # Avoid importing this module at toplevel, as it triggers compilation
        # and can therefore fail
        from numba.cuda.tests.cudapy import recursion_usecases
        self.mod = recursion_usecases
        super().setUp()

    def check_fib(self, cfunc):
        @cuda.jit
        def kernel(r, x):
            r[0] = cfunc(x[0])

        x = np.asarray([10], dtype=np.int64)
        r = np.zeros_like(x)
        kernel[1, 1](r, x)

        actual = r[0]
        expected = 55
        self.assertPreciseEqual(actual, expected)

    def test_global_explicit_sig(self):
        self.check_fib(self.mod.fib1)

    def test_inner_explicit_sig(self):
        self.check_fib(self.mod.fib2)

    def test_global_implicit_sig(self):
        self.check_fib(self.mod.fib3)

    def test_runaway(self):
        with self.assertRaises(TypingError) as raises:
            cfunc = self.mod.runaway_self

            @cuda.jit('void()')
            def kernel():
                cfunc(1)

        self.assertIn("cannot type infer runaway recursion",
                      str(raises.exception))

    @unittest.skip('Needs insert_unresolved_ref support in target')
    def test_type_change(self):
        pfunc = self.mod.type_change_self.py_func
        cfunc = self.mod.type_change_self

        @cuda.jit
        def kernel(r, x, y):
            r[0] = cfunc(x[0], y[0])

        args = 13, 0.125
        x = np.asarray([args[0]], dtype=np.int64)
        y = np.asarray([args[1]], dtype=np.float64)
        r = np.zeros_like(x)

        kernel[1, 1](r, x, y)

        expected = pfunc(*args)
        actual = r[0]

        self.assertPreciseEqual(actual, expected)

    @unittest.expectedFailure
    def test_raise(self):
        # This is an expected failure because reporting of exceptions raised in
        # device functions does not work correctly - see Issue #8036:
        # https://github.com/numba/numba/issues/8036
        with self.assertRaises(ValueError) as raises:
            self.mod.raise_self_kernel[1, 1](3)

        self.assertEqual(str(raises.exception), "raise_self")

    @unittest.skip('Needs insert_unresolved_ref support in target')
    def test_optional_return(self):
        pfunc = self.mod.make_optional_return_case()
        cfunc = self.mod.make_optional_return_case(cuda.jit)

        @cuda.jit
        def kernel(r, x):
            res = cfunc(x[0])
            if res is None:
                res = 999
            r[0] = res

        for arg in (0, 5, 10, 15):

            expected = pfunc(arg)
            if expected is None:
                expected = 999

            x = np.asarray([arg], dtype=np.int64)
            r = np.zeros_like(x)
            kernel[1, 1](r, x)
            actual = r[0]

            self.assertEqual(expected, actual)

    def test_growing_return_tuple(self):
        cfunc = self.mod.make_growing_tuple_case(cuda.jit)

        with self.assertRaises(TypingError) as raises:
            @cuda.jit('void()')
            def kernel():
                cfunc(100)

        self.assertIn(
            "Return type of recursive function does not converge",
            str(raises.exception),
        )


@unittest.skip
class TestMutualRecursion(CUDATestCase):

    def setUp(self):
        from numba.tests import recursion_usecases
        self.mod = recursion_usecases

    def test_mutual_1(self):
        expect = math.factorial(10)
        self.assertPreciseEqual(self.mod.outer_fac(10), expect)

    def test_mutual_2(self):
        pfoo, pbar = self.mod.make_mutual2()
        cfoo, cbar = self.mod.make_mutual2(jit(nopython=True))
        for x in [-1, 0, 1, 3]:
            self.assertPreciseEqual(pfoo(x=x), cfoo(x=x))
            self.assertPreciseEqual(pbar(y=x, z=1), cbar(y=x, z=1))

    def test_runaway(self):
        with self.assertRaises(TypingError) as raises:
            self.mod.runaway_mutual(123)
        self.assertIn("cannot type infer runaway recursion",
                      str(raises.exception))

    def test_type_change(self):
        pfunc = self.mod.make_type_change_mutual()
        cfunc = self.mod.make_type_change_mutual(jit(nopython=True))
        args = 13, 0.125
        self.assertPreciseEqual(pfunc(*args), cfunc(*args))

    def test_four_level(self):
        pfunc = self.mod.make_four_level()
        cfunc = self.mod.make_four_level(jit(nopython=True))
        arg = 7
        self.assertPreciseEqual(pfunc(arg), cfunc(arg))

    def test_inner_error(self):
        # nopython mode
        cfunc = self.mod.make_inner_error(jit(nopython=True))
        with self.assertRaises(TypingError) as raises:
            cfunc(2)
        errmsg = 'Unknown attribute \'ndim\''
        self.assertIn(errmsg, str(raises.exception))
        # objectmode
        # error is never trigger, function return normally
        cfunc = self.mod.make_inner_error(jit)
        pfunc = self.mod.make_inner_error()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NumbaWarning)
            got = cfunc(6)
        self.assertEqual(got, pfunc(6))

    def test_raise(self):
        cfunc = self.mod.make_raise_mutual()#jit(nopython=True))
        with self.assertRaises(ValueError) as raises:
            cfunc(2)

        self.assertEqual(str(raises.exception), "raise_mutual")


if __name__ == '__main__':
    unittest.main()
