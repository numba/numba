from __future__ import print_function

import decimal
import itertools

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types, utils, njit, errors, typeof, numpy_support
from .support import TestCase, tag


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

Noflags = Flags()
Noflags.set("nrt")


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

def slicing_1d_usecase7(a, start):
    # Omitted stop with negative step (issue #1690)
    b = a[start::-2]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def slicing_1d_usecase8(a, start):
    # Omitted start with negative step
    b = a[::-2]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total


def slicing_2d_usecase(a, start1, stop1, step1, start2, stop2, step2):
    # The index is a homogenous tuple of slices
    return a[start1:stop1:step1, start2:stop2:step2]

def slicing_2d_usecase3(a, start1, stop1, step1, index):
    # The index is a heterogenous tuple
    return a[start1:stop1:step1, index]

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

def partial_1d_usecase(a, index):
    b = a[index]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total

def integer_indexing_1d_usecase(a, i):
    return a[i]

def integer_indexing_2d_usecase(a, i1, i2):
    return a[i1,i2]

def integer_indexing_2d_usecase2(a, i1, i2):
    return a[i1][i2]

def ellipsis_usecase1(a, i, j):
    return a[i:j, ...]

def ellipsis_usecase2(a, i, j):
    return a[..., i:j]

def ellipsis_usecase3(a, i, j):
    return a[i, ..., j]

def none_index_usecase(a):
    return a[None]

def empty_tuple_usecase(a):
    return a[()]


@njit
def setitem_usecase(a, index, value):
    a[index] = value

def slicing_1d_usecase_set(a, b, start, stop, step):
    a[start:stop:step] = b
    return a

def slicing_1d_usecase_add(a, b, start, stop):
    # NOTE: uses the ROT_FOUR opcode on Python 2, only on the [start:stop]
    # with inplace operator form.
    a[start:stop] += b
    return a

def slicing_2d_usecase_set(a, b, start, stop, step, start2, stop2, step2):
    a[start:stop:step,start2:stop2:step2] = b
    return a


class TestGetItem(TestCase):
    """
    Test basic indexed load from an array (returning a view or a scalar).
    Note fancy indexing is tested in test_fancy_indexing.
    """

    def test_1d_slicing(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')
        for indices in [(0, 10, 1),
                        (2, 3, 1),
                        (10, 0, 1),
                        (0, 10, -1),
                        (0, 10, 2),
                        (9, 0, -1),
                        (-5, -2, 1),
                        (0, -1, 1),
                        ]:
            expected = pyfunc(a, *indices)
            self.assertPreciseEqual(cfunc(a, *indices), expected)

    def test_1d_slicing_npm(self):
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

    def check_1d_slicing_with_arg(self, pyfunc, flags):
        args = list(range(-9, 10))

        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(10, dtype='i4')
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

    def test_1d_slicing5(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase5
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing5_npm(self):
        self.test_1d_slicing5(flags=Noflags)

    def test_1d_slicing6(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase6
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing6_npm(self):
        self.test_1d_slicing6(flags=Noflags)

    def test_1d_slicing7(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase7
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing7_npm(self):
        self.test_1d_slicing7(flags=Noflags)

    def test_1d_slicing8(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase8
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing8_npm(self):
        self.test_1d_slicing8(flags=Noflags)

    def test_2d_slicing(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c]
        """
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

    def test_2d_slicing_npm(self):
        self.test_2d_slicing(flags=Noflags)

    def test_2d_slicing2(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c, d:e:f]
        """
        # C layout
        pyfunc = slicing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(100, dtype='i4').reshape(10, 10)

        indices = [(0, 10, 1),
                   (2, 3, 1),
                   (10, 0, 1),
                   (0, 10, -1),
                   (0, 10, 2),
                   (10, 0, -1),
                   (9, 0, -2),
                   (-5, -2, 1),
                   (0, -1, 1),
                   ]
        args = [tup1 + tup2
                for (tup1, tup2) in itertools.product(indices, indices)]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

        # Any layout
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]

        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

    def test_2d_slicing2_npm(self):
        self.test_2d_slicing2(flags=Noflags)

    def test_2d_slicing3(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c, d]
        """
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
            (2, 3, 1, 1),
            (10, 0, -1, 8),
            (9, 0, -2, 4),
            (0, 10, 2, 3),
            (0, -1, 3, 1),
        ]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

        # Any layout
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32,
                  types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]

        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

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

    @tag('important')
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

        # Using a 0-d array as integer index
        arraytype = types.Array(types.int32, 1, 'C')
        indextype = types.Array(types.int16, 0, 'C')
        cr = compile_isolated(pyfunc, (arraytype, indextype), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(3, 13, dtype=np.int32)
        for i in (0, 9, -2):
            idx = np.array(i).astype(np.int16)
            assert idx.ndim == 0
            self.assertEqual(pyfunc(a, idx), cfunc(a, idx))

    def test_1d_integer_indexing_npm(self):
        self.test_1d_integer_indexing(flags=Noflags)

    def test_integer_indexing_1d_for_2d(self, flags=enable_pyobj_flags):
        # Test partial (1d) indexing of a 2d array
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertTrue((pyfunc(a, 0) == cfunc(a, 0)).all())
        self.assertTrue((pyfunc(a, 9) == cfunc(a, 9)).all())
        self.assertTrue((pyfunc(a, -1) == cfunc(a, -1)).all())

        arraytype = types.Array(types.int32, 2, 'A')
        cr = compile_isolated(pyfunc, (arraytype, types.int32), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(20, dtype='i4').reshape(5, 4)[::2]
        self.assertPreciseEqual(pyfunc(a, 0), cfunc(a, 0))

    @tag('important')
    def test_integer_indexing_1d_for_2d_npm(self):
        self.test_integer_indexing_1d_for_2d(flags=Noflags)

    def test_2d_integer_indexing(self, flags=enable_pyobj_flags,
                                 pyfunc=integer_indexing_2d_usecase):
        # C layout
        a = np.arange(100, dtype='i4').reshape(10, 10)
        arraytype = types.Array(types.int32, 2, 'C')
        cr = compile_isolated(pyfunc, (arraytype, types.int32, types.int32),
                              flags=flags)
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 3), cfunc(a, 0, 3))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))

        # Any layout
        a = np.arange(100, dtype='i4').reshape(10, 10)[::2, ::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])

        arraytype = types.Array(types.int32, 2, 'A')
        cr = compile_isolated(pyfunc, (arraytype, types.int32, types.int32),
                              flags=flags)
        cfunc = cr.entry_point

        self.assertEqual(pyfunc(a, 0, 1), cfunc(a, 0, 1))
        self.assertEqual(pyfunc(a, 2, 2), cfunc(a, 2, 2))
        self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))

        # With 0-d arrays as integer indices
        a = np.arange(100, dtype='i4').reshape(10, 10)
        arraytype = types.Array(types.int32, 2, 'C')
        indextype = types.Array(types.int32, 0, 'C')
        cr = compile_isolated(pyfunc, (arraytype, indextype, indextype),
                              flags=flags)
        cfunc = cr.entry_point

        for i, j in [(0, 3), (8, 9), (-2, -1)]:
            i = np.array(i).astype(np.int32)
            j = np.array(j).astype(np.int32)
            self.assertEqual(pyfunc(a, i, j), cfunc(a, i, j))

    @tag('important')
    def test_2d_integer_indexing_npm(self):
        self.test_2d_integer_indexing(flags=Noflags)

    def test_2d_integer_indexing2(self):
        self.test_2d_integer_indexing(pyfunc=integer_indexing_2d_usecase2)
        self.test_2d_integer_indexing(flags=Noflags,
                                      pyfunc=integer_indexing_2d_usecase2)

    def test_2d_integer_indexing_via_call(self):
        @njit
        def index1(X, i0):
            return X[i0]
        @njit
        def index2(X, i0, i1):
            return index1(X[i0], i1)
        a = np.arange(10).reshape(2, 5)
        self.assertEqual(index2(a, 0, 0), a[0][0])
        self.assertEqual(index2(a, 1, 1), a[1][1])
        self.assertEqual(index2(a, -1, -1), a[-1][-1])

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

    def test_partial_1d_indexing(self, flags=enable_pyobj_flags):
        pyfunc = partial_1d_usecase

        def check(arr, arraytype):
            cr = compile_isolated(pyfunc, (arraytype, types.int32),
                                  flags=flags)
            cfunc = cr.entry_point
            self.assertEqual(pyfunc(arr, 0), cfunc(arr, 0))
            n = arr.shape[0] - 1
            self.assertEqual(pyfunc(arr, n), cfunc(arr, n))
            self.assertEqual(pyfunc(arr, -1), cfunc(arr, -1))

        a = np.arange(12, dtype='i4').reshape((4, 3))
        arraytype = types.Array(types.int32, 2, 'C')
        check(a, arraytype)

        a = np.arange(12, dtype='i4').reshape((3, 4)).T
        arraytype = types.Array(types.int32, 2, 'F')
        check(a, arraytype)

        a = np.arange(12, dtype='i4').reshape((3, 4))[::2]
        arraytype = types.Array(types.int32, 2, 'A')
        check(a, arraytype)

    def check_ellipsis(self, pyfunc, flags):
        def compile_func(arr):
            cr = compile_isolated(pyfunc, (typeof(arr), types.intp, types.intp),
                                  flags=flags)
            return cr.entry_point

        def run(a):
            bounds = (0, 1, 2, -1, -2)
            cfunc = compile_func(a)
            for i, j in itertools.product(bounds, bounds):
                x = cfunc(a, i, j)
                self.assertPreciseEqual(pyfunc(a, i, j), cfunc(a, i, j))

        run(np.arange(16, dtype='i4').reshape(4, 4))
        run(np.arange(27, dtype='i4').reshape(3, 3, 3))

    def test_ellipsis1(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase1, flags)

    def test_ellipsis1_npm(self):
        self.test_ellipsis1(flags=Noflags)

    def test_ellipsis2(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase2, flags)

    def test_ellipsis2_npm(self):
        self.test_ellipsis2(flags=Noflags)

    def test_ellipsis3(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase3, flags)

    def test_ellipsis3_npm(self):
        self.test_ellipsis3(flags=Noflags)

    def test_ellipsis_issue1498(self):
        # This is an issue due to incorrect layout inferred for when
        # ellpsis is used and ndenumerate is specializing on the layout.
        @njit
        def udt(arr):
            out = np.zeros_like(arr)
            i = 0
            for index, val in np.ndenumerate(arr[..., i]):
                out[index][i] = val

            return out

        py_func = udt.py_func

        outersize = 4
        innersize = 4
        arr = np.arange(outersize * innersize).reshape(outersize, innersize)
        got = udt(arr)
        expected = py_func(arr)
        np.testing.assert_equal(got, expected)

    def test_ellipsis_issue1499(self):
        # This tests an issue when ndarray.__getitem__ recv a tuple of
        # constants. The lowering is mishandling the constant value creation.
        @njit
        def udt(arr):
            return arr[..., 0]

        arr = np.arange(3)
        got = udt(arr)
        expected = udt.py_func(arr)
        np.testing.assert_equal(got, expected)

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

    def test_empty_tuple_indexing(self, flags=enable_pyobj_flags):
        pyfunc = empty_tuple_usecase
        arraytype = types.Array(types.int32, 0, 'C')
        cr = compile_isolated(pyfunc, (arraytype,), flags=flags)
        cfunc = cr.entry_point

        a = np.arange(1, dtype='i4').reshape(())
        self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_empty_tuple_indexing_npm(self):
        self.test_empty_tuple_indexing(flags=Noflags)


class TestSetItem(TestCase):
    """
    Test basic indexed store into an array.
    Note fancy indexing is tested in test_fancy_indexing.
    """

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
        """
        1d to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        # Note heterogenous types for the source and destination arrays
        # (int16[:] -> int32[:])
        dest_type = types.Array(types.int32, 1, 'C')
        src_type = types.Array(types.int16, 1, 'A')
        argtys = (dest_type, src_type, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        N = 10
        arg = np.arange(N, dtype='i2') + 40
        bounds = [0, 2, N - 2, N, N + 1, N + 3,
                  -2, -N + 2, -N, -N - 1, -N - 3]
        def make_dest():
            return np.zeros_like(arg, dtype='i4')
        for start, stop in itertools.product(bounds, bounds):
            for step in (1, 2, -1, -2):
                args = start, stop, step
                index = slice(*args)
                pyleft = pyfunc(make_dest(), arg[index], *args)
                cleft = cfunc(make_dest(), arg[index], *args)
                self.assertPreciseEqual(pyleft, cleft)

        # Mismatching input size and slice length
        with self.assertRaises(ValueError):
            cfunc(np.zeros_like(arg), arg, 0, 0, 1)

    def check_1d_slicing_set_sequence(self, flags, seqty, seq):
        """
        Generic sequence to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        dest_type = types.Array(types.int32, 1, 'C')
        argtys = (dest_type, seqty, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        N = 10
        k = len(seq)
        arg = np.arange(N, dtype=np.int32)
        args = (seq, 1, -N + k + 1, 1)
        expected = pyfunc(arg.copy(), *args)
        got = cfunc(arg.copy(), *args)
        self.assertPreciseEqual(expected, got)

        if numpy_support.version != (1, 7):
            # Numpy 1.7 doesn't always raise an error here (object mode)
            args = (seq, 1, -N + k, 1)
            with self.assertRaises(ValueError) as raises:
                cfunc(arg.copy(), *args)

    def test_1d_slicing_set_tuple(self, flags=enable_pyobj_flags):
        """
        Tuple to 1d slice assignment
        """
        self.check_1d_slicing_set_sequence(
            flags, types.UniTuple(types.int16, 2), (8, -42))

    def test_1d_slicing_set_list(self, flags=enable_pyobj_flags):
        """
        List to 1d slice assignment
        """
        self.check_1d_slicing_set_sequence(
            flags, types.List(types.int16), [8, -42])

    def test_1d_slicing_broadcast(self, flags=enable_pyobj_flags):
        """
        scalar to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        arraytype = types.Array(types.int32, 1, 'C')
        # Note heterogenous types for the source scalar and the destination
        # array (int16 -> int32[:])
        argtys = (arraytype, types.int16, types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        N = 10
        arg = np.arange(N, dtype='i4')
        val = 42
        bounds = [0, 2, N - 2, N, N + 1, N + 3,
                  -2, -N + 2, -N, -N - 1, -N - 3]
        for start, stop in itertools.product(bounds, bounds):
            for step in (1, 2, -1, -2):
                args = val, start, stop, step
                pyleft = pyfunc(arg.copy(), *args)
                cleft = cfunc(arg.copy(), *args)
                self.assertPreciseEqual(pyleft, cleft)

    def test_1d_slicing_add(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase_add
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, arraytype, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        arg = np.arange(10, dtype='i4')
        for test in ((0, 10), (2, 5)):
            pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            cleft = cfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            self.assertTrue((pyleft == cleft).all())

    def test_1d_slicing_set_npm(self):
        self.test_1d_slicing_set(flags=Noflags)

    def test_1d_slicing_set_list_npm(self):
        self.test_1d_slicing_set_list(flags=Noflags)

    def test_1d_slicing_set_tuple_npm(self):
        self.test_1d_slicing_set_tuple(flags=Noflags)

    def test_1d_slicing_broadcast_npm(self):
        self.test_1d_slicing_broadcast(flags=Noflags)

    def test_1d_slicing_add_npm(self):
        self.test_1d_slicing_add(flags=Noflags)

    @tag('important')
    def test_2d_slicing_set(self, flags=enable_pyobj_flags):
        """
        2d to 2d slice assignment
        """
        pyfunc = slicing_2d_usecase_set
        arraytype = types.Array(types.int32, 2, 'A')
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
            self.assertPreciseEqual(cleft, pyleft)

    def test_2d_slicing_broadcast(self, flags=enable_pyobj_flags):
        """
        scalar to 2d slice assignment
        """
        pyfunc = slicing_2d_usecase_set
        arraytype = types.Array(types.int32, 2, 'C')
        # Note heterogenous types for the source scalar and the destination
        # array (int16 -> int32[:])
        argtys = (arraytype, types.int16, types.int32, types.int32, types.int32,
                  types.int32, types.int32, types.int32)
        cr = compile_isolated(pyfunc, argtys, flags=flags)
        cfunc = cr.entry_point

        arg = np.arange(10*10, dtype='i4').reshape(10,10)
        val = 42
        tests = [
            (0, 10, 1, 0, 10, 1),
            (2, 3, 1, 2, 3, 1),
            (10, 0, 1, 10, 0, 1),
            (0, 10, -1, 0, 10, -1),
            (0, 10, 2, 0, 10, 2),
        ]
        for test in tests:
            pyleft = pyfunc(arg.copy(), val, *test)
            cleft = cfunc(arg.copy(), val, *test)
            self.assertPreciseEqual(cleft, pyleft)

    def test_2d_slicing_set_npm(self):
        self.test_2d_slicing_set(flags=Noflags)

    def test_2d_slicing_broadcast_npm(self):
        self.test_2d_slicing_broadcast(flags=Noflags)

    def test_setitem(self):
        """
        scalar indexed assignment
        """
        arr = np.arange(5)
        setitem_usecase(arr, 1, 42)
        self.assertEqual(arr.tolist(), [0, 42, 2, 3, 4])
        # Using a 0-d array as scalar index
        setitem_usecase(arr, np.array(3).astype(np.uint16), 8)
        self.assertEqual(arr.tolist(), [0, 42, 2, 8, 4])
        # Broadcasting
        arr = np.arange(9).reshape(3, 3)
        setitem_usecase(arr, 1, 42)
        self.assertEqual(arr.tolist(), [[0, 1, 2], [42, 42, 42], [6, 7, 8]])

    def test_setitem_readonly(self):
        arr = np.arange(5)
        arr.flags.writeable = False
        with self.assertRaises((TypeError, errors.TypingError)) as raises:
            setitem_usecase(arr, 1, 42)
        self.assertIn("Cannot modify value of type readonly array",
                      str(raises.exception))


class TestTyping(TestCase):
    """
    Check typing of basic indexing operations
    """

    def test_layout(self):
        """
        Check an appropriate layout is inferred for the result of array
        indexing.
        """
        from numba.typing import arraydecl
        from numba.types import intp, ellipsis, slice2_type, slice3_type

        func = arraydecl.get_array_index_type

        cty = types.Array(types.float64, 3, 'C')
        fty = types.Array(types.float64, 3, 'F')
        aty = types.Array(types.float64, 3, 'A')

        indices = [
            # Tuples of (indexing arguments, keeps "C" layout, keeps "F" layout)
            ((), True, True),
            ((ellipsis,), True, True),

            # Indexing from the left => can sometimes keep "C" layout
            ((intp,), True, False),
            ((slice2_type,), True, False),
            ((intp, slice2_type), True, False),
            ((slice2_type, intp), False, False),
            ((slice2_type, slice2_type), False, False),
            # Strided slices = > "A" layout
            ((intp, slice3_type), False, False),
            ((slice3_type,), False, False),

            # Indexing from the right => can sometimes keep "F" layout
            ((ellipsis, intp,), False, True),
            ((ellipsis, slice2_type,), False, True),
            ((ellipsis, intp, slice2_type,), False, False),
            ((ellipsis, slice2_type, intp,), False, True),
            ((ellipsis, slice2_type, slice2_type,), False, False),
            # Strided slices = > "A" layout
            ((ellipsis, slice3_type,), False, False),
            ((ellipsis, slice3_type, intp,), False, False),

            # Indexing from both sides => only if all dimensions are indexed
            ((intp, ellipsis, intp,), False, False),
            ((slice2_type, ellipsis, slice2_type,), False, False),
            ((intp, intp, slice2_type,), True, False),
            ((intp, ellipsis, intp, slice2_type,), True, False),
            ((slice2_type, intp, intp,), False, True),
            ((slice2_type, intp, ellipsis, intp,), False, True),
            ((intp, slice2_type, intp,), False, False),
            # Strided slices = > "A" layout
            ((slice3_type, intp, intp,), False, False),
            ((intp, intp, slice3_type,), False, False),
            ]

        for index_tuple, keep_c, _ in indices:
            index = types.Tuple(index_tuple)
            r = func(cty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'C' if keep_c else 'A',
                             index_tuple)
            self.assertFalse(r.advanced)

        for index_tuple, _, keep_f in indices:
            index = types.Tuple(index_tuple)
            r = func(fty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'F' if keep_f else 'A',
                             index_tuple)
            self.assertFalse(r.advanced)

        for index_tuple, _, _ in indices:
            index = types.Tuple(index_tuple)
            r = func(aty, index)
            self.assertEqual(tuple(r.index), index_tuple)
            self.assertEqual(r.result.layout, 'A')
            self.assertFalse(r.advanced)


if __name__ == '__main__':
    unittest.main()
