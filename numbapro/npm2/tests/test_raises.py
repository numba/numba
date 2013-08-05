import dis
import numpy as np
from ..compiler import compile
from ..types import (arraytype, float32, void)
from .support import testcase, main

value_is_negative = ValueError("value is negative")

def raise_if_neg(val):
    if val < 0:
        raise value_is_negative

@testcase
def test_raises_1():
    cfunc = compile(raise_if_neg, void, [float32])
    cfunc(10)

    try:
        cfunc(-10)
    except ValueError, e:
        print e
        assert e is value_is_negative
    else:
        raise AssertionError('should raise exception')

if __name__ == '__main__':
    main()
