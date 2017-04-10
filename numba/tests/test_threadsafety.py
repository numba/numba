"""
Test threadsafety for compiler.
These tests will cause segfault if fail.
"""
import threading
import random

import numpy as np

from numba import config
from numba import unittest_support as unittest
from numba import jit, vectorize, guvectorize

from .support import temp_directory, override_config


class TestThreadSafety(unittest.TestCase):

    def run_jit(self, **options):
        def runner():
            @jit(**options)
            def foo(n, v):
                return np.ones(n)
            return foo(4, 10)
        return runner

    def run_compile(self, fnlist):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
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
        self.run_compile([self.run_jit(nopython=True)])

    def test_concurrent_jit_cache(self):
        self.run_compile([self.run_jit(nopython=True, cache=True)])

    def run_vectorize(self, **options):
        def runner():
            @vectorize(['(f4, f4)'], **options)
            def foo(a, b):
                return a + b

            a = b = np.random.random(10).astype(np.float32)
            return foo(a, b)
        return runner

    def test_concurrent_vectorize(self):
        self.run_compile([self.run_vectorize(nopython=True)])

    def test_concurrent_vectorize_cache(self):
        self.run_compile([self.run_vectorize(nopython=True, cache=True)])

    def run_guvectorize(self, **options):
        def runner():
            @guvectorize(['(f4, f4, f4[:])'], '(),()->()', **options)
            def foo(a, b, out):
                out[0] = a + b

            a = b = np.random.random(10).astype(np.float32)
            return foo(a, b)
        return runner

    def test_concurrent_guvectorize(self):
        self.run_compile([self.run_guvectorize(nopython=True)])

    def test_concurrent_guvectorize_cache(self):
        self.run_compile([self.run_guvectorize(nopython=True, cache=True)])

    def test_concurrent_mix_use(self):
        self.run_compile([self.run_jit(nopython=True, cache=True),
                          self.run_jit(nopython=True),
                          self.run_vectorize(nopython=True, cache=True),
                          self.run_vectorize(nopython=True),
                          self.run_guvectorize(nopython=True, cache=True),
                          self.run_guvectorize(nopython=True)])


if __name__ == '__main__':
    unittest.main()
