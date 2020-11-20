from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin


@jit
def inc(a):
    for i in range(len(a)):
        a[i] += 1
    return a

@jit
def inc1(a):
    a[0] += 1
    return a[0]

@jit
def inc2(a):
    a[0] += 1
    return a[0], a[0] + 1


def chain1(a):
    x = y = z = inc(a)
    return x + y + z


def chain2(v):
    a = np.zeros(2)
    a[0] = x = a[1] = v
    return a[0] + a[1] + (x / 2)


def unpack1(x, y):
    a, b = x, y
    return a + b / 2


def unpack2(x, y):
    a, b = c, d = inc1(x), inc1(y)
    return a + c / 2, b + d / 2


def chain3(x, y):
    a = (b, c) = (inc1(x), inc1(y))
    (d, e) = f = (inc1(x), inc1(y))
    return (a[0] + b / 2 + d + f[0]), (a[1] + c + e / 2 + f[1])


def unpack3(x):
    a, b = inc2(x)
    return a + b / 2


def unpack4(x):
    a, b = c, d = inc2(x)
    return a + c / 2, b + d / 2


def unpack5(x):
    a = b, c = inc2(x)
    d, e = f = inc2(x)
    return (a[0] + b / 2 + d + f[0]), (a[1] + c + e / 2 + f[1])


def unpack6(x, y):
    (a, b), (c, d) = (x, y), (y + 1, x + 1)
    return a + c / 2, b / 2 + d


class TestChainedAssign(MemoryLeakMixin, unittest.TestCase):
    def test_chain1(self):
        args = [
            [np.arange(2)],
            [np.arange(4, dtype=np.double)],
        ]
        self._test_template(chain1, args)

    def test_chain2(self):
        args = [
            [3],
            [3.0],
        ]
        self._test_template(chain2, args)

    def test_unpack1(self):
        args = [
            [1, 3.0],
            [1.0, 3],
        ]
        self._test_template(unpack1, args)

    def test_unpack2(self):
        args = [
            [np.array([2]), np.array([4.0])],
            [np.array([2.0]), np.array([4])],
        ]
        self._test_template(unpack2, args)

    def test_chain3(self):
        args = [
            [np.array([0]), np.array([1.5])],
            [np.array([0.5]), np.array([1])],
        ]
        self._test_template(chain3, args)

    def test_unpack3(self):
        args = [
            [np.array([1])],
            [np.array([1.0])],
        ]
        self._test_template(unpack3, args)

    def test_unpack4(self):
        args = [
            [np.array([1])],
            [np.array([1.0])],
        ]
        self._test_template(unpack4, args)

    def test_unpack5(self):
        args = [
            [np.array([2])],
            [np.array([2.0])],
        ]
        self._test_template(unpack5, args)

    def test_unpack6(self):
        args1 = 3.0, 2
        args2 = 3.0, 2.0
        self._test_template(unpack6, [args1, args2])

    def _test_template(self, pyfunc, argcases):
        cfunc = jit(pyfunc)
        for args in argcases:
            a1 = copy.deepcopy(args)
            a2 = copy.deepcopy(args)
            np.testing.assert_allclose(pyfunc(*a1), cfunc(*a2))


if __name__ == '__main__':
    unittest.main()

