from contextlib import contextmanager
from ..compiler import compile
from ..types import (
    int8, int32, int64, uint8, uint16, uint32, float64
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

def caster_template(sty, dty, arg, exc):
    compiled = compile(caster, dty, [sty])
    with assert_raise(exc):
        compiled(arg)


@testcase
def test_signed_to_unsigned():
    caster_template(int32, uint32, -123, OverflowError)

@testcase
def test_signed_to_signed_truncated():
    caster_template(int32, int8, -256, OverflowError)

@testcase
def test_unsigned_to_signed():
    caster_template(uint8, int8, 128, OverflowError)

@testcase
def test_unsigned_to_signed_truncated():
    caster_template(uint16, int8, 0xffff, OverflowError)

@testcase
def test_nan_to_int():
    caster_template(float64, int64, float('nan'), ValueError)

if __name__ == '__main__':
    main()
