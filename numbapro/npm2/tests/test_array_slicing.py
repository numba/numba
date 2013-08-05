import numpy as np
from ..compiler import compile
from ..types import (arraytype, float32)
from .support import testcase, main

def array_slicing_1(ary):
    slice = ary[:]      # entire array
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

if __name__ == '__main__':
    main()
