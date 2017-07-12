from __future__ import print_function, division, absolute_import

import math
import numpy as np
from numba import vectorize, guvectorize, njit, types
from .support import unittest


@vectorize
def jitted_v1(alpha, beta):
    return np.sin(alpha) + np.cos(beta)


sig = [(types.float32[:], types.float32[:], types.float32[:]),
       (types.float64[:], types.float64[:], types.float64[:])]


@guvectorize(sig, '(n),()->(n)')
def jitted_g1(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y[0]


@njit
def jitted_j1(x, y, alpha, beta):
    acc = 0
    for k in range(len(x)):
        acc += x[k] - np.sum(y) + alpha - beta
    return acc


@njit
def jitted_j2(x):
    return x + 1


class TestCallChains(unittest.TestCase):
    """
    Tests this matrix of decorators
    +-------------+------+-----------+-------------+
    |             | njit | vectorize | guvectorize |
    +-------------+------+-----------+-------------+
    | njit        |  ok  |     ok    |      ok     |
    | vectorize   |  ok  |     ok    | unsupported |
    | guvectorize |  ok  |     ok    |      ok     |
    +-------------+------+-----------+-------------+
    """

    def __init__(self, *args):
        self.n = 10
        super(TestCallChains, self).__init__(*args)

    def jit_args(self):
        alpha = 2.
        beta = 3. + 1j
        x = np.arange(self.n, dtype=np.float64)
        y = np.arange(2 * self.n, dtype=np.float64)[::-2]
        return (x, y, alpha, beta)

    def vect_args(self, amount=3):
        x = np.arange(self.n, dtype=np.float64)
        y = np.arange(2 * self.n, dtype=np.float64)[::-2]
        z = np.zeros_like(x)
        return (x, y, z)[:amount]

    def check_guvectorize(self, pyfunc, spec):
        cfunc = guvectorize(sig, spec)(pyfunc)

        x1, y1, z1 = self.vect_args()
        pyfunc(x1, y1, z1)
        expected = z1

        x2, y2, z2 = self.vect_args()
        z2 = cfunc(x2, y2, z2)
        got = z2

        np.testing.assert_allclose(expected, got)

    def check_jitted(self, pyfunc, decorator, arg_func, **kwargs):
        cfunc = decorator(pyfunc)

        expected = pyfunc(*arg_func(**kwargs))
        got = cfunc(*arg_func(**kwargs))

        np.testing.assert_allclose(expected, got)

    def test_njit_call_vectorize(self):

        def func(x, y, alpha, beta):
            acc = 0
            for k in range(len(x)):
                acc += x[k] - np.sum(jitted_v1(x, beta + y))
            return acc

        self.check_jitted(func, njit, self.jit_args)

    def test_vectorize_call_njit(self):

        def func(alpha, beta):
            return np.sin(alpha) + jitted_j2(beta)

        self.check_jitted(func, vectorize, self.vect_args, amount=2)

    def test_vectorize_call_vectorize(self):

        def func(alpha, beta):
            return np.sin(alpha) + jitted_v1(alpha, beta)

        self.check_jitted(func, vectorize, self.vect_args, amount=2)

    def test_njit_call_guvectorize(self):

        def func(x, y):
            z = np.zeros_like(x)
            jitted_g1(x, y, z)
            return z

        self.check_jitted(func, njit, self.vect_args, amount=2)

    def test_guvectorize_call_njit(self):

        def func(x, y, res):
            for i in range(x.shape[0]):
                res[i] = jitted_j2(x[i]) + y[0]

        self.check_guvectorize(func, '(n), ()->(n)')

    def test_guvectorize_call_vectorize(self):

        def func(x, y, res):
            res[:] = jitted_v1(x, y)

        self.check_guvectorize(func, '(n), (n)->(n)')

    def test_njit_call_njit(self):

        def func(x, y, alpha, beta):
            acc = 0
            for k in range(len(x)):
                acc += x[k] - np.abs(jitted_j1(x, y, alpha, beta))
            return acc

        self.check_jitted(func, njit, self.jit_args)

    def test_guvectorize_call_guvectorize(self):

        def func(x, y, res):
            jitted_g1(x, y, res)
            for i in range(x.shape[0]):
                res[i] = x[i] + y[0]

        self.check_guvectorize(func, '(n), ()->(n)')
