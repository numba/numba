import sys
import numpy as np

#docutils = py.test.importorskip("theano")
import unittest

try:
    import theano
    raise ImportError
except ImportError:
    raise unittest.SkipTest

from numba.ad import Watcher


def sqr(x):
    y = 2 * x  # irrelevant
    return x * x


def compute_stuff(x):
    a = np.ones(3) + x
    b = 2 * a
    c = np.sum(sqr(b))
    print 'HELLO', c   # <-- way easier to do this than in Theano
    return c


def test_foo():
    x = np.zeros(3)

    w = Watcher([x])
    y = w.call(compute_stuff, x)
    assert y == 12

    f = w.recalculate_fn(y, x)
    assert f(x) == 12
    assert f(x + 1) != 12


def test_grad():
    x = np.zeros(3)

    w = Watcher([x])
    assert id(x) in w.svars
    y = w.call(compute_stuff, x)
    assert id(y) in w.svars
    dy_dx_fn = w.grad_fn(y, x)

    print dy_dx_fn(x)
    print dy_dx_fn(x + 1)
    print dy_dx_fn(x + 2)


@unittest.skip("AttributeError: 'FrameVM' object has no attribute "
               "'op_POP_JUMP_IF_TRUE'")
def test_loop():

    def repeat_double(x, N):
        print 'N', N
        for i in range(N):
            x = x + x
            print 'i', i, 'x', x
        assert i == N - 1
        return x

    #repeat_double(0, 4)

    x = np.zeros(3)
    w = Watcher([x])
    y = w.call(repeat_double, x, 4)

    f = w.recalculate_fn(y, x)
    y2 = f(x + 1)
    assert np.all(y == 0)
    assert np.all(y2 == 16)

if __name__ == "__main__":
#    import __main__
#    import numba
#    numba.nose_run(module=__main__)
    test_foo()
    test_grad()
#    test_loop()
