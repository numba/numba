from contextlib import contextmanager
from ..compiler import compile
from ..types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, complex64, complex128
)
from .support import testcase, main

def caster(a):
    return a

@contextmanager
def assert_raise(exc):
    try:
        yield
    except exc, e:
        print e
    else:
        raise AssertionError('expecting exception: %s' % exc)

@testcase
def test_signed_to_unsigned():
    compiled = compile(caster, uint32, [int32])
    with assert_raise(OverflowError):
        compiled(-123)

@testcase
def test_signed_to_signed_truncated():
    compiled = compile(caster, int8, [int32])
    with assert_raise(OverflowError):
        compiled(-256)

if __name__ == '__main__':
    main()
