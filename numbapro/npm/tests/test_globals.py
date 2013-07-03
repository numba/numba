from ..compiler import compile
from ..types import int32, float64, complex128
from .support import testcase, main

A_INT = 1234
A_FLOAT = 123.432
A_COMPLEX = 12+43j

def global_int():
    return A_INT

def global_float():
    return A_FLOAT

def global_complex():
    return A_COMPLEX

@testcase
def test_global_int():
    compiled = compile(global_int, int32, [])
    got = compiled()
    exp = global_int()
    assert got == exp, (got, exp)

@testcase
def test_global_float():
    compiled = compile(global_float, float64, [])
    got = compiled()
    exp = global_float()
    assert got == exp, (got, exp)

@testcase
def test_global_complex():
    compiled = compile(global_complex, complex128, [])
    got = compiled()
    exp = global_complex()
    assert got == exp, (got, exp)

if __name__ == '__main__':
    main()
