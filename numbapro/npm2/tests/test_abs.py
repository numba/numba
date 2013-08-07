import numpy as np
from ..compiler import compile
from ..types import int8, int32, float32
from .support import testcase, main

def myabs(a):
    return abs(a)

@testcase
def test_myabs_integer():
    compiled = compile(myabs, int32, [int32])
    args = [-123, 123]
    for a in args:
        exp = myabs(a)
        got = compiled(a)
        assert got == exp, (got, exp)

@testcase
def test_myabs_integer_overflow():
    compiled = compile(myabs, int8, [int8])
    try:
        got = compiled(-128)
    except OverflowError, e:
        print e
    else:
        raise AssertionError("expecting exception")

@testcase
def test_myabs_float():
    compiled = compile(myabs, float32, [float32])
    args = [-12.3, 12.3]
    for a in args:
        exp = myabs(a)
        got = compiled(a)
        assert np.allclose(got, exp), (got, exp)

if __name__ == '__main__':
    main()
