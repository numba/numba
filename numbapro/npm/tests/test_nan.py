import numpy as np
from ..compiler import compile
from ..types import boolean, float32
from .support import testcase, main, assertTrue

def fne(a, b):
    return a != b

def feq(a, b):
    return a == b


@testcase
def test_feq():
    compiled = compile(feq, boolean, [float32, float32])
    a, b = float('nan'), float('nan')
    got = compiled(a, b)
    exp = feq(a, b)
    assertTrue(got == exp, (got, exp))

@testcase
def test_fne():
    compiled = compile(fne, boolean, [float32, float32])
    a, b = float('nan'), float('nan')
    got = compiled(a, b)
    exp = fne(a, b)
    assertTrue(got == exp, (got, exp))


if __name__ == '__main__':
    main()
