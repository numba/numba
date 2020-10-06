#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


class TestNumpy_math_functions(DPPLTestCase):
    N = 10
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)

    def test_add(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.add(a, b)
            return c

        c = f(self.a, self.b)
        d = self.a + self.b
        self.assertTrue(np.all(c == d))

    def test_subtract(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.subtract(a, b)
            return c

        c = f(self.a, self.b)
        d = self.a - self.b
        self.assertTrue(np.all(c == d))

    def test_multiply(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.multiply(a, b)
            return c

        c = f(self.a, self.b)
        d = self.a * self.b
        self.assertTrue(np.all(c == d))

    def test_divide(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.divide(a, b)
            return c

        c = f(self.a, self.b)
        d = self.a / self.b
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_true_divide(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.true_divide(a, b)
            return c

        c = f(self.a, self.b)
        d = np.true_divide(self.a, self.b)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_negative(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.negative(a)
            return c

        c = f(self.a)
        self.assertTrue(np.all(c == -self.a))

    def test_power(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.power(a, b)
            return c

        input_arr = np.random.randint(self.N, size=(self.N))
        exp = np.full((self.N), 2, dtype=np.int)

        c = f(input_arr, exp)
        self.assertTrue(np.all(c == input_arr * input_arr))

    def test_remainder(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.remainder(a, b)
            return c

        input_arr = np.full((self.N), 3, dtype=np.int)
        divisor = np.full((self.N), 2, dtype=np.int)

        c = f(input_arr, divisor)
        self.assertTrue(np.all(c == 1))

    def test_mod(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.mod(a, b)
            return c

        input_arr = np.full((self.N), 3, dtype=np.int)
        divisor = np.full((self.N), 2, dtype=np.int)

        c = f(input_arr, divisor)
        self.assertTrue(np.all(c == 1))

    def test_fmod(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.fmod(a, b)
            return c

        input_arr = np.full((self.N), 3, dtype=np.float32)
        divisor = np.full((self.N), 2, dtype=np.int)

        c = f(input_arr, divisor)
        self.assertTrue(np.all(c == 1.))

    def test_abs(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.abs(a)
            return c

        input_arr = 5 * np.random.random_sample(self.N) - 5

        c = f(input_arr)
        self.assertTrue(np.all(c == -input_arr))

    def test_absolute(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.absolute(a)
            return c

        input_arr = 5 * np.random.random_sample(self.N) - 5

        c = f(input_arr)
        self.assertTrue(np.all(c == -input_arr))


    def test_fabs(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.fabs(a)
            return c

        input_arr = 5 * np.random.random_sample(self.N) - 5

        c = f(input_arr)
        self.assertTrue(np.all(c == -input_arr))


    def test_sign(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sign(a)
            return c

        input_arr = 5 * np.random.random_sample(self.N) - 5

        c = f(input_arr)
        self.assertTrue(np.all(c == -1.))

    def test_conj(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.conj(a)
            return c

        input_arr = np.eye(self.N) + 1j * np.eye(self.N)

        c = f(input_arr)
        d = np.conj(input_arr)
        self.assertTrue(np.all(c == d))

    def test_exp(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.exp(a)
            return c

        input_arr = np.random.randint(self.N, size=(self.N))
        c = f(input_arr)
        d = np.exp(input_arr)
        self.assertTrue(np.all(c == d))


    def test_log(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.log(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))
        c = f(input_arr)
        d = np.log(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_log10(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.log10(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))
        c = f(input_arr)
        d = np.log10(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_expm1(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.expm1(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))
        c = f(input_arr)
        d = np.expm1(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_log1p(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.log1p(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))
        c = f(input_arr)
        d = np.log1p(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_sqrt(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sqrt(a)
            return c

        c = f(self.a)
        d = np.sqrt(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_square(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.square(a)
            return c

        input_arr = np.random.randint(self.N, size=(self.N))

        c = f(input_arr)
        self.assertTrue(np.all(c == input_arr * input_arr))

    def test_reciprocal(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.reciprocal(a)
            return c

        input_arr =  5 * np.random.random_sample(self.N) + 5

        c = f(input_arr)
        self.assertTrue(np.all(c == 1/input_arr))

    def test_conjugate(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.conjugate(a)
            return c

        input_arr = np.eye(self.N) + 1j * np.eye(self.N)

        c = f(input_arr)
        d = np.conj(input_arr)
        self.assertTrue(np.all(c == d))


if __name__ == '__main__':
    unittest.main()
