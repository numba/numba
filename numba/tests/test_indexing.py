from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases
from .support import TestCase

import decimal


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

Noflags = Flags()


def slicing_1d_usecase(a, start, stop, step):
    return a[start:stop:step]

def slicing_1d_usecase2(a, start, stop, step):
    b = a[start:stop:step]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_1d_usecase3(a, start, stop):
    b = a[start:stop]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_1d_usecase4(a):
    b = a[:]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_1d_usecase5(a, start):
    b = a[start:]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_1d_usecase6(a, stop):
    b = a[:stop]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_2d_usecase(a, start1, stop1, step1, start2, stop2, step2):
    return a[start1:stop1:step1, start2:stop2:step2]

def slicing_2d_usecase2(a, start1, stop1, step1, start2, stop2, step2):
    b = a[start1:stop1:step1, start2:stop2:step2]
    total = 0
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            total += b[i, j] * (i + j + 1)
    return total

def slicing_2d_usecase3(a, start1, stop1, step1, index):
    b = a[start1:stop1:step1, index]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_3d_usecase(a, index0, start1, index2):
    b = a[index0, start1:, index2]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_3d_usecase2(a, index0, stop1, index2):
    b = a[index0, :stop1, index2]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

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

def empty_tuple_usecase(a):
    return a[()]


def slicing_1d_usecase_set(a, b, start, stop, step):
    a[start:stop:step] = b
    return a

def slicing_2d_usecase_set(a, b, start, stop, step, start2, stop2, step2):
    a[start:stop:step,start2:stop2:step2] = b
    return a


class TestIndexing(TestCase):

    def test_1d_slicing(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')
        self.assertTrue((pyfunc(a, 0, 10, 1) == cfunc(a, 0, 10, 1)).all())
        self.assertTrue((pyfunc(a, 2, 3, 1) == cfunc(a, 2, 3, 1)).all())
        self.assertTrue((pyfunc(a, 10, 0, 1) == cfunc(a, 10, 0, 1)).all())
        self.assertTrue((pyfunc(a, 0, 10, -1) == cfunc(a, 0, 10, -1)).all())
        self.assertTrue((pyfunc(a, 0, 10, 2) == cfunc(a, 0, 10, 2)).all())

    def test_1d_slicing_npm(self):
        """
        Return of arbitrary array is not supported yet
        """
        with self.assertTypingError():
            self.test_1d_slicing(flags=Noflags)

    def test_1d_slicing2(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase2
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')

        args = [(0, 10, 1),
                (2, 3, 1),
                (10, 0, 1),
                (0, 10, -1),
                (0, 10, 2)]

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))


        # Any
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])

        args = [(0, 10, 1),
                (2, 3, 1),
                (10, 0, 1),
                (0, 10, -1),
                (0, 10, 2)]

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_1d_slicing2_npm(self):
        self.test_1d_slicing2(flags=Noflags)

    def test_1d_slicing3(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase3
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')

        args = [(3, 10),
                (2, 3),
                (10, 0),
                (0, 10),
                (5, 10)]

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))


        # Any
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_1d_slicing3_npm(self):
        self.test_1d_slicing3(flags=Noflags)

    def test_1d_slicing4(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase4
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype,)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')
        self.assertEqual(pyfunc(a), cfunc(a))

        # Any
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype,)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        self.assertEqual(pyfunc(a), cfunc(a))

    def test_1d_slicing4_npm(self):
        self.test_1d_slicing4(flags=Noflags)

    def test_1d_slicing5(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase5
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')

        args = [3, 2, 10, 0, 5]

        for arg in args:
            self.assertEqual(pyfunc(a, arg), cfunc(a, arg))


        # Any
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        for arg in args:
            self.assertEqual(pyfunc(a, arg), cfunc(a, arg))

    def test_1d_slicing5_npm(self):
        self.test_1d_slicing5(flags=Noflags)

    def test_2d_slicing(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
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
        cr = compile_isolated(pyfunc, argtys, flags=flags)
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

    def test_2d_slicing_npm(self):
        with self.assertTypingError():
            self.test_2d_slicing(flags=Noflags)

    def test_2d_slicing2(self, flags=enable_pyobj_flags):
        # C layout
        pyfunc = slicing_2d_usecase2
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(100, dtype='i4').reshape(10, 10)

        args = [
            (0, 10, 1, 0, 10, 1),
            (2, 3, 1, 2, 3, 1),
            (10, 0, 1, 10, 0, 1),
            (0, 10, -1, 0, 10, -1),
            (0, 10, 2, 0, 10, 2),
        ]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

        # Any layout
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_2d_slicing2_npm(self):
        self.test_2d_slicing2(flags=Noflags)

    def test_2d_slicing3(self, flags=enable_pyobj_flags):
        # C layout
        pyfunc = slicing_2d_usecase3
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(100, dtype='i4').reshape(10, 10)

        args = [
            (0, 10, 1, 0),
            (2, 3, 1, 2),
            (10, 0, 1, 9),
            (0, 10, -1, 0),
            (0, 10, 2, 4),
        ]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

        # Any layout
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_2d_slicing3_npm(self):
        self.test_2d_slicing3(flags=Noflags)

    def test_3d_slicing(self, flags=enable_pyobj_flags):
        # C layout
        pyfunc = slicing_3d_usecase
        arraytype = types.Array(types.int32, 3, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(1000, dtype='i4').reshape(10, 10, 10)

        args = [
            (0, 9, 1),
            (2, 3, 1),
            (9, 0, 1),
            (0, 9, -1),
            (0, 9, 2),
        ]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

        # Any layout
        arraytype = types.Array(types.int32, 3, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(2000, dtype='i4')[::2].reshape(10, 10, 10)

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_3d_slicing_npm(self):
        self.test_3d_slicing(flags=Noflags)

    def test_3d_slicing2(self, flags=enable_pyobj_flags):
        # C layout
        pyfunc = slicing_3d_usecase2
        arraytype = types.Array(types.int32, 3, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(1000, dtype='i4').reshape(10, 10, 10)

        args = [
            (0, 9, 1),
            (2, 3, 1),
            (9, 0, 1),
            (0, 9, -1),
            (0, 9, 2),
        ]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

        # Any layout
        arraytype = types.Array(types.int32, 3, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(2000, dtype='i4')[::2].reshape(10, 10, 10)

        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_3d_slicing2_npm(self):
        self.test_3d_slicing2(flags=Noflags)

    def test_1d_integer_indexing(self, flags=enable_pyobj_flags):
        # C layout
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32), flags=flags)
        cfunc = cr.entry_point
        
        a = np.arange(10, dtype='i4')
        self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertEqual(pyfunc(a, 9), cfunc(a, 9))
        self.assertEqual(pyfunc(a, -1), cfunc(a, -1))

        # Any layout
        arraytype = types.Array(types.int32, 1, 'A')
        cr = compile_isolated(pyfunc, (arraytype, types.int32), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertEqual(pyfunc(a, 2), cfunc(a, 2))
        self.assertEqual(pyfunc(a, -1), cfunc(a, -1))

    def test_1d_integer_indexing_npm(self):
        self.test_1d_integer_indexing(flags=Noflags)

    def test_integer_indexing_1d_for_2d(self, flags=enable_pyobj_flags):
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32), flags=flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a, 0) == cfunc(a, 0)).all())
        self.assertTrue((pyfunc(a, 9) == cfunc(a, 9)).all())
        self.assertTrue((pyfunc(a, -1) == cfunc(a, -1)).all())


    def test_integer_indexing_1d_for_2d(self):
        with self.assertTypingError():
            self.test_integer_indexing_1d_for_2d(flags=Noflags)

    def test_2d_integer_indexing(self, flags=enable_pyobj_flags):
        # C layout
        a = np.arange(100, dtype='i4').reshape(10, 10)
        pyfunc = integer_indexing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32, types.int32),
                              flags=flags)
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))

        # Any layout
        a = np.arange(100, dtype='i4').reshape(10, 10)[::2, ::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])

        pyfunc = integer_indexing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'A')
        cr = compile_isolated(pyfunc, (arraytype, types.int32, types.int32),
                              flags=flags)
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
        self.assertEqual(pyfunc(a, 2, 2), cfunc(a, 2, 2))
        self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))

    def test_2d_integer_indexing_npm(self):
        self.test_2d_integer_indexing(flags=Noflags)

    def test_2d_float_indexing(self, flags=enable_pyobj_flags):
        a = np.arange(100, dtype='i4').reshape(10, 10)
        pyfunc = integer_indexing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.float32, types.int32),
                              flags=flags)
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))

    def test_ellipse(self, flags=enable_pyobj_flags):
        pyfunc = ellipse_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        # TODO should be enable to handle this in NoPython mode
        cr = compile_isolated(pyfunc, (arraytype,), flags=flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a) == cfunc(a)).all())

    def test_ellipse_npm(self):
        with self.assertTypingError():
            self.test_ellipse(flags=Noflags)

    def test_none_index(self, flags=enable_pyobj_flags):
        pyfunc = none_index_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        # TODO should be enable to handle this in NoPython mode
        cr = compile_isolated(pyfunc, (arraytype,), flags=flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a) == cfunc(a)).all())

    def test_none_index_npm(self):
        with self.assertTypingError():
            self.test_none_index(flags=Noflags)

    def test_fancy_index(self, flags=enable_pyobj_flags):
        pyfunc = fancy_index_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        indextype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, indextype), flags=flags)
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

    def test_fancy_index_npm(self):
        with self.assertTypingError():
            self.test_fancy_index(flags=Noflags)

    def test_boolean_indexing(self, flags=enable_pyobj_flags):
        pyfunc = boolean_indexing_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        masktype = types.Array(types.boolean, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype, masktype), flags=flags)
        cfunc = cr.entry_point
        
        a = np.arange(100, dtype='i4').reshape(10, 10)
        mask = np.array([True, False, True])
        self.assertTrue((pyfunc(a, mask) == cfunc(a, mask)).all())

    def test_boolean_indexing_npm(self):
        with self.assertTypingError():
            self.test_boolean_indexing(flags=Noflags)

    def test_empty_tuple_indexing(self, flags=enable_pyobj_flags):
        pyfunc = empty_tuple_usecase
        arraytype = types.Array(types.int32, 0, 'C')
        cr = compile_isolated(pyfunc, (arraytype,), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(1, dtype='i4').reshape(())
        self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_empty_tuple_indexing_npm(self):
        self.test_empty_tuple_indexing(flags=Noflags)

    def test_conversion_setitem(self, flags=enable_pyobj_flags):
        """ this used to work, and was used in one of the tutorials """
        from numba import jit

        def pyfunc(array):
            for index in range(len(array)):
                array[index] = index % decimal.Decimal(100)

        cfunc = jit("void(i8[:])")(pyfunc)

        udt = np.arange(100, dtype='i1')
        control = udt.copy()
        pyfunc(control)
        cfunc(udt)
        self.assertTrue((udt == control).all())

    def test_1d_slicing_set(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase_set
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        arg = np.arange(10, dtype='i4')
        for test in ((0, 10, 1), (2,3,1), (10,0,1), (0,10,-1), (0,10,2)):
            pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            cleft = cfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            self.assertTrue((pyleft == cleft).all())

    def test_1d_slicing_set_npm(self):
        """
        TypingError: Cannot resolve setitem: array(int32, 1d, C)[slice3_type] = ...
        setitem on slices not yet supported.
        """
        with self.assertTypingError():
            self.test_1d_slicing_set(flags=Noflags)

    def test_2d_slicing_set(self, flags=enable_pyobj_flags):
        pyfunc = slicing_2d_usecase_set
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        arg = np.arange(10*10, dtype='i4').reshape(10,10)
        tests = [
            (0, 10, 1, 0, 10, 1),
            (2, 3, 1, 2, 3, 1),
            (10, 0, 1, 10, 0, 1),
            (0, 10, -1, 0, 10, -1),
            (0, 10, 2, 0, 10, 2),
        ]
        for test in tests:
            pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test[0:3]), slice(*test[3:6])], *test)
            cleft = cfunc(np.zeros_like(arg), arg[slice(*test[0:3]), slice(*test[3:6])], *test)
            self.assertTrue((pyleft == cleft).all())

    def test_2d_slicing_set_npm(self):
        """
        TypingError: TypingError: Cannot resolve setitem: array(int32, 2d, C)[(slice3_type x 2)] = array(int32, 2d, C)
        setitem on slices not yet supported.
        """
        with self.assertTypingError():
            self.test_2d_slicing_set(flags=Noflags)


if __name__ == '__main__':
    unittest.main()

