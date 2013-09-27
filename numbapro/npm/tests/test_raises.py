from ..compiler import compile
from ..types import float32, void
from .support import testcase, main, assertTrue

value_is_negative = ValueError("value is negative")

def raise_if_neg(val):
    if val < 0:
        raise value_is_negative

def raise_if_neg_class(val):
    if val < 0:
        raise ValueError

@testcase
def test_raises_1():
    cfunc = compile(raise_if_neg, void, [float32])
    cfunc(10)

    try:
        cfunc(-10)
    except ValueError, e:
        print e
        assertTrue(e is value_is_negative)
    else:
        raise AssertionError('should raise exception')


@testcase
def test_raises_2():
    cfunc = compile(raise_if_neg_class, void, [float32])
    cfunc(10)

    try:
        cfunc(-10)
    except ValueError, e:
        assertTrue(type(e) is ValueError)
    else:
        raise AssertionError('should raise exception')

if __name__ == '__main__':
    main()
