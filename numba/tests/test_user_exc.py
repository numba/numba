from numba.compiler import compile_isolated, Flags
from numba import types
from numba import unittest_support as unittest
from numba.pythonapi import NativeError
import numpy as np


class MyError(Exception):
    pass


def pyfunc(i):
    if i:
        raise MyError


class TestUserExc(unittest.TestCase):
    def test_unituple_index_error(self):
        def pyfunc(a, i):
            return a.shape[i]

        cres = compile_isolated(pyfunc, (types.Array(types.int32, 1, 'A'),
                                         types.int32))

        cfunc = cres.entry_point
        a = np.empty(2)

        self.assertEqual(cfunc(a, 0), pyfunc(a, 0))

        with self.assertRaises(NativeError):
            cfunc(a, 2)

    def test_raise_nopython(self):
        args = [types.int32]
        cres = compile_isolated(pyfunc, args)
        cfunc = cres.entry_point
        cfunc(0)

        with self.assertRaises(MyError):
            cfunc(1)

    def test_raise_object(self):
        args = [types.int32]
        flags = Flags()
        flags.set('force_pyobject')
        cres = compile_isolated(pyfunc, args,
                                flags=flags)
        cfunc = cres.entry_point
        cfunc(0)

        with self.assertRaises(MyError):
            cfunc(1)

if __name__ == '__main__':
    unittest.main()
