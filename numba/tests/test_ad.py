import numpy as np

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


