import numpy as np

from numba.ad import CallVM, Watcher


def sqr(x):
    y = 2 * x
    return x * x


def compute_stuff(x):
    a = np.ones(3) + x
    b = 2 * a
    c = sqr(b).sum()
    return c


def test_foo():

    x = np.zeros(3)

    w = Watcher([x])
    y = w.call(compute_stuff, x)
    assert y == 12


def test_grad():
    x = np.zeros(3)

    w = Watcher([x])
    y = w.call(compute_stuff, x)

    dy_dx_fn = Watcher.grad_fn(y, x)

    print dy_dx(x)
    print dy_dx(x + 1)
    print dy_dx(x + 2)

if __name__ == '__main__':
    sys.exit(main())
