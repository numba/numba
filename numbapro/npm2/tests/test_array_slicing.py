import numpy as np
from ..compiler import compile
from ..types import (arraytype, float32, int32)
from .support import testcase, main

def array_slicing_1(ary):
    slice = ary[:]                  # entire array
    tmp = 0
    for i in range(slice.shape[0]):
        tmp += slice[i]
    return tmp

def array_slicing_2(ary):
    slice = ary[:ary.shape[0]//2]   # 1st half
    tmp = 0
    for i in range(slice.shape[0]):
        tmp += slice[i]
    return tmp

def array_slicing_3(ary):
    slice = ary[ary.shape[0]//2:]   # 2nd half
    tmp = 0
    for i in range(slice.shape[0]):
        tmp += slice[i]
    return tmp

def array_slicing_4(ary, a, b):
    slice = ary[a:b]                # sandwiche
    tmp = 0
    for i in range(slice.shape[0]):
        tmp += slice[i]
    return tmp

def array_slicing_2d_1(ary, a, b):
    slice = ary[a:, b:]
    tmp = 0
    for i in range(slice.shape[0]):
        for j in range(slice.shape[1]):
            tmp += slice[i, j]
    return tmp

@testcase
def test_array_slicing_1():
    cfunc = compile(array_slicing_1, float32, [arraytype(float32, 1, 'C')])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a), array_slicing_1(a))

@testcase
def test_array_slicing_2():
    cfunc = compile(array_slicing_2, float32, [arraytype(float32, 1, 'C')])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a), array_slicing_2(a))

@testcase
def test_array_slicing_3():
    cfunc = compile(array_slicing_3, float32, [arraytype(float32, 1, 'C')])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a), array_slicing_3(a))

@testcase
def test_array_slicing_4():
    cfunc = compile(array_slicing_4, float32,
                                     [arraytype(float32, 1, 'C'), int32, int32])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a, 3, 7), array_slicing_4(a, 3, 7))

@testcase
def test_array_slicing_5():
    "test wraparound"
    cfunc = compile(array_slicing_4, float32,
                                     [arraytype(float32, 1, 'C'), int32, int32])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a, 3, -1), array_slicing_4(a, 3, -1))

@testcase
def test_array_slicing_6():
    "test clipping"
    cfunc = compile(array_slicing_4, float32,
                                     [arraytype(float32, 1, 'C'), int32, int32])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a, 3, 11), array_slicing_4(a, 3, 11))

@testcase
def test_array_slicing_7():
    "test wraparound"
    cfunc = compile(array_slicing_4, float32,
                                     [arraytype(float32, 1, 'C'), int32, int32])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a, -4, -1), array_slicing_4(a, -4, -1))

@testcase
def test_array_slicing_2d_1_C():
    cfunc = compile(array_slicing_2d_1, float32, [arraytype(float32, 2, 'C'),
                                                  int32, int32])
    a = np.arange(40, dtype=np.float32).reshape(8, 5)
    assert np.allclose(cfunc(a, 2, 3), array_slicing_2d_1(a, 2, 3))

@testcase
def test_array_slicing_2d_1_F():
    cfunc = compile(array_slicing_2d_1, float32, [arraytype(float32, 2, 'F'),
                                                  int32, int32])
    a = np.arange(40, dtype=np.float32).reshape(8, 5, order='F')
    assert np.allclose(cfunc(a, 2, 3), array_slicing_2d_1(a, 2, 3))



if __name__ == '__main__':
    main()
