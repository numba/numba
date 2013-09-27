import numpy as np
from ..compiler import compile
from ..types import int32, float32
from .support import testcase, main, assertTrue

def min2(a, b):
    return min(a, b)

def max2(a, b):
    return max(a, b)

def min3(a, b, c):
    return min(a, b, c)

def max3(a, b, c):
    return max(a, b, c)

#-------------------------------------------------------------------------------
# min2

@testcase
def test_min2_integer():
    compiled = compile(min2, int32, [int32, int32])
    a, b = 321, 123
    got = compiled(a, b)
    exp = min2(a, b)
    assertTrue(got == exp, (got, exp))

@testcase
def test_min2_float():
    compiled = compile(min2, float32, [float32, float32])
    a, b = 32.1, 12.3
    got = compiled(a, b)
    exp = min2(a, b)
    assertTrue(np.allclose(got, exp), (got, exp))

#-------------------------------------------------------------------------------
# max2

@testcase
def test_max2_integer():
    compiled = compile(max2, int32, [int32, int32])
    a, b = 321, 123
    got = compiled(a, b)
    exp = max2(a, b)
    assertTrue(got == exp, (got, exp))

@testcase
def test_max2_float():
    compiled = compile(max2, float32, [float32, float32])
    a, b = 32.1, 12.3
    got = compiled(a, b)
    exp = max2(a, b)
    assertTrue(np.allclose(got, exp), (got, exp))

#-------------------------------------------------------------------------------
# min3

@testcase
def test_min3_integer():
    compiled = compile(min3, int32, [int32, int32, int32])
    a, b, c = 321, 123, 91
    got = compiled(a, b, c)
    exp = min3(a, b, c)
    assertTrue(got == exp, (got, exp))

@testcase
def test_min3_float():
    compiled = compile(min3, float32, [float32, float32, float32])
    a, b, c = 32.1, 12.3, 2091.
    got = compiled(a, b, c)
    exp = min3(a, b, c)
    assertTrue(np.allclose(got, exp), (got, exp))

#-------------------------------------------------------------------------------
# max3

@testcase
def test_max3_integer():
    compiled = compile(max3, int32, [int32, int32, int32])
    a, b, c = 321, 123, 91
    got = compiled(a, b, c)
    exp = max3(a, b, c)
    assertTrue(got == exp, (got, exp))

@testcase
def test_max3_float():
    compiled = compile(max3, float32, [float32, float32, float32])
    a, b, c = 32.1, 12.3, 2091.
    got = compiled(a, b, c)
    exp = max3(a, b, c)
    assertTrue(np.allclose(got, exp), (got, exp))


if __name__ == '__main__':
    main()
