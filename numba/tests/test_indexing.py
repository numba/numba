from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases

import decimal


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def slicing_1d_usecase(a, start, stop, step):
    return a[start:stop:step]

def slicing_2d_usecase(a, start1, stop1, step1, start2, stop2, step2):
    return a[start1:stop1:step1,start2:stop2:step2]

def integer_indexing_1d_usecase(a, i):
    return a[i]

def integer_indexing_2d_usecase(a, i1, i2):
    return a[i1,i2]

def ellipse_usecase(a):
    return a[...,0]

def none_index_usecase(a):
    return a[None]

def fancy_index_usecase(a, index):
    return a[index]

def boolean_indexing_usecase(a, mask):
    return a[mask]


class TestIndexing(unittest.TestCase):

    def test_1d_slicing(self):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')
        self.assertTrue((pyfunc(a, 0, 10, 1) == cfunc(a, 0, 10, 1)).all())
        self.assertTrue((pyfunc(a, 2, 3, 1) == cfunc(a, 2, 3, 1)).all())
        self.assertTrue((pyfunc(a, 10, 0, 1) == cfunc(a, 10, 0, 1)).all())
        self.assertTrue((pyfunc(a, 0, 10, -1) == cfunc(a, 0, 10, -1)).all())
        self.assertTrue((pyfunc(a, 0, 10, 2) == cfunc(a, 0, 10, 2)).all())

    def test_2d_slicing(self):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a, 0, 10, 1) == cfunc(a, 0, 10, 1)).all())
        self.assertTrue((pyfunc(a, 2, 3, 1) == cfunc(a, 2, 3, 1)).all())
        self.assertTrue((pyfunc(a, 10, 0, 1) == cfunc(a, 10, 0, 1)).all())
        self.assertTrue((pyfunc(a, 0, 10, -1) == cfunc(a, 0, 10, -1)).all())
        self.assertTrue((pyfunc(a, 0, 10, 2) == cfunc(a, 0, 10, 2)).all())

        pyfunc = slicing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=enable_pyobj_flags)
        cfunc = cr.entry_point

        self.assertTrue((pyfunc(a, 0, 10, 1, 0, 10, 1) ==
                         cfunc(a, 0, 10, 1, 0, 10, 1)).all())
        self.assertTrue((pyfunc(a, 2, 3, 1, 2, 3, 1) ==
                         cfunc(a, 2, 3, 1, 2, 3, 1)).all())
        self.assertTrue((pyfunc(a, 10, 0, 1, 10, 0, 1) ==
                         cfunc(a, 10, 0, 1, 10, 0, 1)).all())
        self.assertTrue((pyfunc(a, 0, 10, -1, 0, 10, -1) ==
                         cfunc(a, 0, 10, -1, 0, 10, -1)).all())
        self.assertTrue((pyfunc(a, 0, 10, 2, 0, 10, 2) ==
                         cfunc(a, 0, 10, 2, 0, 10, 2)).all())

    def test_1d_integer_indexing(self):
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32))
        cfunc = cr.entry_point
        
        a = np.arange(10, dtype='i4')
        self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertEqual(pyfunc(a, 9), cfunc(a, 9))
        self.assertEqual(pyfunc(a, -1), cfunc(a, -1))

    def test_2d_integer_indexing(self):
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a, 0) == cfunc(a, 0)).all())
        self.assertTrue((pyfunc(a, 9) == cfunc(a, 9)).all())
        self.assertTrue((pyfunc(a, -1) == cfunc(a, -1)).all())

        pyfunc = integer_indexing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32, types.int32))
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))

    def test_ellipse(self):
        pyfunc = ellipse_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        # TODO should be enable to handle this in NoPython mode
        cr = compile_isolated(pyfunc, (arraytype,), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a) == cfunc(a)).all())

    def test_none_index(self):
        pyfunc = none_index_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        # TODO should be enable to handle this in NoPython mode
        cr = compile_isolated(pyfunc, (arraytype,), flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a) == cfunc(a)).all())

    def test_fancy_index(self):
        pyfunc = fancy_index_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        indextype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, indextype),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        index = np.array([], dtype='i4')
        self.assertTrue((pyfunc(a, index) == cfunc(a, index)).all())
        index = np.array([0], dtype='i4')
        self.assertTrue((pyfunc(a, index) == cfunc(a, index)).all())
        index = np.array([1,2], dtype='i4')
        self.assertTrue((pyfunc(a, index) == cfunc(a, index)).all())
        index = np.array([-1], dtype='i4')
        self.assertTrue((pyfunc(a, index) == cfunc(a, index)).all())

    def test_boolean_indexing(self):
        pyfunc = boolean_indexing_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        masktype = types.Array(types.boolean, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, masktype),
                              flags=enable_pyobj_flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        mask = np.array([True, False, True])
        self.assertTrue((pyfunc(a, mask) == cfunc(a, mask)).all())

    def test_conversion_setitem(self):
        """ this used to work, and was used in one of the tutorials """
        from numba import jit

        def pyfunc(array):
            for index in xrange(len(array)):
                array[index] = index % decimal.Decimal(100)

        cfunc = jit("void(i8[:])")(pyfunc)

        a = np.arange(100, dtype='i1')
        self.assertTrue((pyfunc(a, 1, 42) == cfunc(a,1,42)).all())


if __name__ == '__main__':
    unittest.main()

