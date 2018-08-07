"""
Tests the parallel backend
"""
import threading
import random

import numpy as np

from numba import config
from numba import unittest_support as unittest
from numba import jit, vectorize, guvectorize

def foo(n, v):
    return np.ones(n)

def ufunc_foo(a, b):
    return a + b

def gufunc_foo(a, b, out):
    out[0] = a + b

_threadsafe_backends = config.THREADING_LAYER in ('omppool', 'tbbpool')

@unittest.skipUnless(_threadsafe_backends, "Threading layer not threadsafe")
class TestParallelBackend(unittest.TestCase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend
    """

    def run_jit(self, **options):
        def runner():
            cfunc = jit(**options)(foo)

            return cfunc(4, 10)
        return runner

    def run_compile(self, fnlist):
        def chooser():
            for _ in range(10):
                fn = random.choice(fnlist)
                fn()

        ths = [threading.Thread(target=chooser)
                for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()

    def test_concurrent_jit(self):
        self.run_compile([self.run_jit(nopython=True, parallel=True)])

    def run_vectorize(self, **options):
        def runner():
            cfunc = vectorize(['(f4, f4)'], **options)(ufunc_foo)
            a = b = np.random.random(10).astype(np.float32)
            return cfunc(a, b)
        return runner

    def test_concurrent_vectorize(self):
        self.run_compile([self.run_vectorize(nopython=True, target='parallel')])

    def run_guvectorize(self, **options):
        def runner():
            sig = ['(f4, f4, f4[:])']
            cfunc = guvectorize(sig, '(),()->()', **options)(gufunc_foo)
            a = b = np.random.random(10).astype(np.float32)
            return cfunc(a, b)
        return runner

    def test_concurrent_guvectorize(self):
        self.run_compile([self.run_guvectorize(nopython=True,
                                               target='parallel')])

    def test_concurrent_mix_use(self):
        self.run_compile([self.run_jit(nopython=True),
                          self.run_jit(nopython=True, nogil=True),
                          self.run_jit(nopython=True, parallel=True),
                          self.run_vectorize(nopython=True),
                          self.run_vectorize(nopython=True, target='parallel'),
                          self.run_guvectorize(nopython=True),
                          self.run_guvectorize(nopython=True, target='parallel'),
                          ])


# TODO: Test fork() behaviours, and fork() cascade, and then fork(threads()) and
# threads(fork()), then then random permutations of.

if __name__ == '__main__':
    unittest.main()
