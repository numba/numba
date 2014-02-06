from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types


def array_shape(a, i):
    return a.shape[i]


def array_strides(a, i):
    return a.strides[i]


def array_ndim(a):
    return a.ndim


def array_size(a):
    return a.size


class TestArrayAttr(unittest.TestCase):
    def test_shape(self):
        pyfunc = array_shape
        cres = compile_isolated(pyfunc, (types.int32[:,:], types.int32))
        cfunc = cres.entry_point

        a = np.arange(10).reshape(2, 5)
        for i in range(a.ndim):
            self.assertEqual(pyfunc(a, i), cfunc(a, i))

    def test_strides(self):
        pyfunc = array_strides
        cres = compile_isolated(pyfunc, (types.int32[:,:], types.int32))
        cfunc = cres.entry_point

        a = np.arange(10).reshape(2, 5)
        for i in range(a.ndim):
            self.assertEqual(pyfunc(a, i), cfunc(a, i))

    def test_ndim(self):
        pyfunc = array_ndim
        cres = compile_isolated(pyfunc, (types.int32[:,:],))
        cfunc = cres.entry_point

        a = np.arange(10).reshape(2, 5)
        self.assertEqual(pyfunc(a), cfunc(a))

    def test_size(self):
        pyfunc = array_size
        cres = compile_isolated(pyfunc, (types.int32[:,:],))
        cfunc = cres.entry_point

        a = np.arange(10).reshape(2, 5)
        self.assertEqual(pyfunc(a), cfunc(a))

if __name__ == '__main__':
    unittest.main()

