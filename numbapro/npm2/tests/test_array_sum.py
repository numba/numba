import numpy as np
from ..compiler import compile
from ..types import (arraytype, float32)

from .support import testcase, main

def array_sum_1(ary):
    return ary.sum()

def array_sum_2(ary):
    return np.sum(ary)
#
#@testcase
#def test_array_sum_1():
#    cfunc = compile(array_sum_1, float32, [arraytype(float32, 1, 'C')])
#    a = np.arange(10, dtype=np.float32)
#    print cfunc(a)

@testcase
def test_array_sum_2():
    cfunc = compile(array_sum_2, float32, [arraytype(float32, 1, 'C')])
    a = np.arange(10, dtype=np.float32)
    assert np.allclose(cfunc(a), np.sum(a))

@testcase
def test_array_sum_3():
    cfunc = compile(array_sum_2, float32, [arraytype(float32, 2, 'C')])
    a = np.arange(10, dtype=np.float32).reshape(2, 5)
    assert np.allclose(cfunc(a), np.sum(a))

if __name__ == '__main__':
    main()

