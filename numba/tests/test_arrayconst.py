from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated
from numba.typeinfer import TypingError
from numba import types

myarray = np.arange(5)
myscalar = np.int32(64)


def use_array_const(i):
    return myarray[i]


def use_arrayscalar_const():
    return myscalar


def write_to_global_array():
    myarray[0] = 1


class TestConstantArray(unittest.TestCase):
    def test_array_const(self):
        pyfunc = use_array_const
        cres = compile_isolated(pyfunc, (types.int32,))
        cfunc = cres.entry_point
        for i in [0, 1, 2]:
            self.assertEqual(pyfunc(i), cfunc(i))

    def test_arrayscalar_const(self):
        pyfunc = use_arrayscalar_const
        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point

        self.assertEqual(pyfunc(), cfunc())

    def test_write_to_global_array(self):
        pyfunc = write_to_global_array
        with self.assertRaises(TypingError):
            compile_isolated(pyfunc, ())


if __name__ == '__main__':
    unittest.main()

