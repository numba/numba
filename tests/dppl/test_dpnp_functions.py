#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


def test_for_different_datatypes(fn, test_fn, dims, arg_count, tys, np_all=False, matrix=None):
    if arg_count == 1:
        for ty in tys:
            if matrix and matrix[0]:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty).reshape(dims[0], dims[1])
            else:
                a = np.array(np.random.random(dims[0]), dtype=ty)
            c = fn(a)
            d = test_fn(a)
            if np_all:
                max_abs_err = np.all(c - d)
            else:
                max_abs_err = c - d
            if not (max_abs_err < 1e-4):
                return False

    elif arg_count == 2:
        for ty in tys:
            if matrix and matrix[0]:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty).reshape(dims[0], dims[1])
            else:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty)
            if matrix and matrix[1]:
                b = np.array(np.random.random(dims[2] * dims[3]), dtype=ty).reshape(dims[2], dims[3])
            else:
                b = np.array(np.random.random(dims[2] * dims[3]), dtype=ty)

            c = fn(a, b)
            d = test_fn(a, b)
            if np_all:
                max_abs_err = np.sum(c - d)
            else:
                max_abs_err = c - d
            if not (max_abs_err < 1e-4):
                return False

    return True

def test_for_dimensions(fn, test_fn, dims, tys, np_all=False):
    total_size = 1
    for d in dims:
        total_size *= d

    for ty in tys:
        a = np.array(np.random.random(total_size), dtype=ty).reshape(dims)
        c = fn(a)
        d = test_fn(a)
        if np_all:
            max_abs_err = np.all(c - d)
        else:
            max_abs_err = c - d
        if not (max_abs_err < 1e-4):
            return False

    return True

def ensure_dpnp():
    try:
       # import dpnp
        from numba.dppl.dpnp_glue import dpnp_fptr_interface as dpnp_glue
        return True
    except:
        return False


@unittest.skipUnless(ensure_dpnp(), 'test only when dpNP is available')
class Testdpnp_functions(DPPLTestCase):
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

    def test_sum(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sum(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.sum, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2, 3], self.tys))

    def test_prod(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.prod(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.prod, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2, 3], self.tys))

    def test_argmax(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.argmax(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmax, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmax, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmax, [10, 2, 3], self.tys))

    def test_max(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.max(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.max, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2, 3], self.tys))

    def test_argmin(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.argmin(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmin, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmin, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmin, [10, 2, 3], self.tys))

    def test_min(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.min(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.min, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_argsort(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.argsort(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmin, [10], 1, self.tys, np_all=True))

    def test_median(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.median(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.median, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.median, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.median, [10, 2, 3], self.tys))

    def test_mean(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.mean(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.mean, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2, 3], self.tys))

    def test_matmul(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.matmul(a, b)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.matmul, [10, 5, 5, 10], 2, [np.float, np.double], np_all=True, matrix=[True, True]))

    def test_dot(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.dot(a, b)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.dot, [10, 1, 10, 1], 2, [np.float, np.double]))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [10, 1, 10, 2], 2, [np.float, np.double], matrix=[False, True], np_all=True))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [2, 10, 10, 1], 2, [np.float, np.double], matrix=[True, False], np_all=True))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [10, 2, 2, 10], 2, [np.float, np.double], matrix=[True, True], np_all=True))


    def test_cov(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.cov(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.cov, [10, 7], 1, self.tys, matrix=[True], np_all=True))

    def test_dpnp_interacting_with_parfor(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            #d = a + 1
            return 0

        result = f(self.a, self.b)
        #np_result = np.add((self.a + np.sum(self.a)), self.b)

        #max_abs_err = result.sum() - np_result.sum()
        #self.assertTrue(max_abs_err < 1e-4)


if __name__ == '__main__':
    unittest.main()
