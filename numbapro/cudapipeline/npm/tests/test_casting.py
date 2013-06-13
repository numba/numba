import numpy as np
from ..compiler import compile
from ..types import *
from .support import testcase, main

def cast_int(a):
    return int(a)

def cast_float(a):
    return float(a)

@testcase
def test_cast_int():
    a = 12.53
    compiled = compile(cast_int, int32, [float32])
    exp = cast_int(a)
    got = compiled(a)
    assert got == exp, (got, exp)

@testcase
def test_cast_float():
    a = 1231
    compiled = compile(cast_float, float64, [int32])
    exp = cast_float(a)
    got = compiled(a)
    assert np.allclose(got, exp), (got, exp)

if __name__ == '__main__':
    main()
