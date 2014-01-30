"""
Testing object mode specifics.

"""
from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags


def complex_constant(n):
    tmp = n + 4
    return tmp + 3j


forceobj = Flags()
forceobj.set("force_pyobject")


class TestObjectMode(unittest.TestCase):
    def test_complex_constant(self):
        pyfunc = complex_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(12), cfunc(12))

if __name__ == '__main__':
    unittest.main()
