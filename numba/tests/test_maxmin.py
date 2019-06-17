from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def domax3(a, b, c):
    return max(a, b, c)


def domin3(a, b, c):
    return min(a, b, c)


class TestMaxMin(unittest.TestCase):
    def test_max3(self):
        pyfunc = domax3
        argtys = (types.int32, types.float32, types.double)
        cres = compile_isolated(pyfunc, argtys)
        cfunc = cres.entry_point

        a = 1
        b = 2
        c = 3

        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))

    def test_min3(self):
        pyfunc = domin3
        argtys = (types.int32, types.float32, types.double)
        cres = compile_isolated(pyfunc, argtys)
        cfunc = cres.entry_point

        a = 1
        b = 2
        c = 3

        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))


if __name__ == '__main__':
    unittest.main()
