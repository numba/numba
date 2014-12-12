import numba.unittest_support as unittest

import math
import sys

from numba import jit, utils
from .support import TestCase


max_uint64 = 18446744073709551615

def usecase_uint64_global():
    return max_uint64

def usecase_uint64_constant():
    return 18446744073709551615

def usecase_uint64_floor():
    # This function will only work due to LLVM optimization
    # The following code converts max i64 to f64.
    # Round-to-even will increase the value.
    # When converted back to i64, it can no longer fit into 64-bit.
    # As a result, zero is returned.
    return math.floor(18446744073709551615)


class IntWidthTest(TestCase):

    def check_nullary_func(self, pyfunc, **kwargs):
        cfunc = jit(**kwargs)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_global_uint64(self, nopython=False):
        pyfunc = usecase_uint64_global
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_global_uint64_npm(self):
        self.test_global_uint64(nopython=True)

    def test_constant_uint64(self, nopython=False):
        pyfunc = usecase_uint64_constant
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_npm(self):
        self.test_constant_uint64(nopython=True)

    def test_constant_uint64_function_call(self, nopython=False):
        pyfunc = usecase_uint64_floor
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_function_call_npm(self):
        self.test_constant_uint64_function_call(nopython=True)

    def test_bit_length(self):
        f = utils.bit_length
        self.assertEqual(f(0x7f), 7)
        self.assertEqual(f(-0x7f), 7)
        self.assertEqual(f(0x80), 8)
        self.assertEqual(f(-0x80), 8)
        self.assertEqual(f(0xff), 8)
        self.assertEqual(f(-0xff), 8)
        self.assertEqual(f(0x100), 9)
        self.assertEqual(f(-0x100), 9)
        self.assertEqual(f(0x7fffffff), 31)
        self.assertEqual(f(0x80000000), 32)
        self.assertEqual(f(0xffffffff), 32)
        self.assertEqual(f(0xffffffffffffffff), 64)
        self.assertEqual(f(0x10000000000000000), 65)
        if utils.PYVERSION < (3, 0):
            self.assertEqual(f(long(0xffffffffffffffff)), 64)


if __name__ == '__main__':
    unittest.main()

