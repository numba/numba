import ctypes
import unittest
from collections import namedtuple
from functools import partial
import numba
from numba import *
from numba import ndarray_helpers
import llvm.core as lc

# ______________________________________________________________________

ArrayType = numba.struct([('data', double.pointer()),
                          ('shape', int64.pointer()),
                          ('strides', int64.pointer())])

Int32 = lc.Type.int(32)
const = partial(lc.Constant.int, Int32)
zero = const(0)
one  = const(1)
two  = const(2)

def ptr_at(builder, ptr, idx):
    return builder.gep(ptr, [const(idx)])

def load_at(builder, ptr, idx):
    return builder.load(ptr_at(builder, ptr, idx))

def store_at(builder, ptr, idx, val):
    builder.store(val, ptr_at(builder, ptr, idx))

class MyArray(object):
    """
    Internal array class for a double(:, 10) array.
    """

    def __init__(self, array_ptr, builder):
        self.array_ptr = array_ptr
        self.builder = builder
        self.nd = 2

    @classmethod
    def from_type(cls, llvm_dtype):
        return ArrayType.pointer().to_llvm()

    @property
    def data(self):
        dptr = self.builder.gep(self.array_ptr, [zero, zero])
        return self.builder.load(dptr)

    @property
    def shape_ptr(self):
        result = self.builder.gep(self.array_ptr, [zero, one])
        result = self.builder.load(result)
        return result

    @property
    def strides_ptr(self):
        result = self.builder.gep(self.array_ptr, [zero, two])
        return self.builder.load(result)

    @property
    def shape(self):
        return self.preload(self.shape_ptr, self.nd)

    @property
    def strides(self):
        return self.preload(self.strides_ptr, self.nd)

    @property
    def ndim(self):
        return const(self.nd)

    @property
    def itemsize(self):
        raise NotImplementedError

    def preload(self, ptr, count=None):
        assert count is not None
        return [load_at(self.builder, ptr, i) for i in range(count)]

    def getptr(self, *indices):
        const = partial(lc.Constant.int, indices[0].type)
        offset = self.builder.add(
            self.builder.mul(indices[0], const(10)), indices[1])
        data_ptr_ty = lc.Type.pointer(lc.Type.double())
        dptr_plus_offset = self.builder.gep(self.data, [offset])
        return self.builder.bitcast(dptr_plus_offset, data_ptr_ty)

ndarray_helpers.Array.register(MyArray)

# ______________________________________________________________________
# Test functions

CtypesArray = ArrayType.to_ctypes()

@jit(void(double[:, :]), array=MyArray, wrap=False, nopython=True)
def use_array(A):
    """simple test function"""
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = i * A.shape[1] + j

@jit(object_(double[:, :]), array=MyArray, wrap=False)
def get_attributes(A):
    return A.shape[0], A.shape[1], A.strides[0], A.strides[1]

# ______________________________________________________________________

# Ctypes functions
c_use_array      = numba.addressof(use_array)
c_get_attributes = numba.addressof(get_attributes)

c_use_array.argtypes      = [ctypes.POINTER(CtypesArray)]
c_get_attributes.argtypes = [ctypes.POINTER(CtypesArray)]

# ______________________________________________________________________
# Utils

Array = namedtuple('Array', ['handle', 'array', 'data', 'shape', 'strides'])

def make_array():
    """Make a double[*, 10] ctypes-allocated array"""
    empty = lambda c_type, args: ctypes.cast(
        (c_type * len(args))(*args), ctypes.POINTER(c_type))

    data    = empty(ctypes.c_double, [0] * 50)
    shape   = empty(ctypes.c_int64,  [5, 10])
    strides = empty(ctypes.c_int64,  [10 * 8, 8]) # doubles!
    array   = CtypesArray(data, shape, strides)

    return Array(ctypes.pointer(array), array, data, shape, strides)

# ______________________________________________________________________
# Tests...

class TestArray(unittest.TestCase):

    def test_indexing(self):
        arr = make_array()
        c_use_array(arr.handle)
        for i in range(50):
            assert arr.data[i] == float(i), (arr.data[i], i)

    def test_attributes(self):
        arr = make_array()
        result = c_get_attributes(arr.handle)
        assert result == (5, 10, 80, 8), result


if __name__ == "__main__":
    # TestArray('test_attributes').debug()
    # TestArray('test_indexing').debug()
    unittest.main()
