"""
Testing object mode specifics.

"""
from __future__ import print_function
import numpy
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import utils, jit

def complex_constant(n):
    tmp = n + 4
    return tmp + 3j


forceobj = Flags()
forceobj.set("force_pyobject")


def loop_nest_3(x, y):
    n = 0
    for i in range(x):
        for j in range(y):
            for k in range(x+y):
                n += i * j

    return n


def array_of_object(x):
    return x


class TestObjectMode(unittest.TestCase):
    def test_complex_constant(self):
        pyfunc = complex_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(12), cfunc(12))

    def test_loop_nest(self):
        """
        Test bug that decref the iterator early.
        If the bug occurs, a segfault should occur
        """
        pyfunc = loop_nest_3
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(5, 5), cfunc(5, 5))

        def bm_pyfunc():
            pyfunc(5, 5)

        def bm_cfunc():
            cfunc(5, 5)

        print(utils.benchmark(bm_pyfunc))
        print(utils.benchmark(bm_cfunc))

    def test_array_of_object(self):
        cfunc = jit(array_of_object)
        objarr = numpy.array([object()] * 10)
        self.assertIs(cfunc(objarr), objarr)

if __name__ == '__main__':
    unittest.main()
