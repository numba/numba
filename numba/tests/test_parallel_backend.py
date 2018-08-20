"""
Tests the parallel backend
"""
import threading
import multiprocessing
import random

import numpy as np

from numba import config
from numba import unittest_support as unittest
from numba import jit, vectorize, guvectorize

from .support import temp_directory, override_config

# some functions to jit


def foo(n, v):
    return np.ones(n) + v


def linalg(n, v):
    np.random.seed(42)
    return np.linalg.cond(np.random.random((10, 10))) + np.ones(n) + v


def ufunc_foo(a, b):
    return a + b


def gufunc_foo(a, b, out):
    out[0] = a + b


class runnable(object):
    def __init__(self, **options):
        self._options = options


class jit_runner(runnable):

    def __call__(self):
        cfunc = jit(**self._options)(foo)
        a = 4
        b = 10
        expected = foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)


class linalg_runner(runnable):

    def __call__(self):
        cfunc = jit(**self._options)(linalg)
        a = 4
        b = 10
        expected = linalg(a, b)
        got = cfunc(a, b)
        # broken, fork safe?
        # np.testing.assert_allclose(expected, got)


class vectorize_runner(runnable):

    def __call__(self):
        cfunc = vectorize(['(f4, f4)'], **self._options)(ufunc_foo)
        a = b = np.random.random(10).astype(np.float32)
        expected = ufunc_foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)


class guvectorize_runner(runnable):

    def __call__(self):
        def runner():
            sig = ['(f4, f4, f4[:])']
            cfunc = guvectorize(sig, '(),()->()', **self._options)(gufunc_foo)
            a = b = np.random.random(10).astype(np.float32)
            expected = ufunc_foo(a, b)
            got = cfunc(a, b)
            np.testing.assert_allclose(expected, got)
        return runner


def chooser(fnlist):
    for _ in range(10):
        fn = random.choice(fnlist)
        fn()


def compile_factory(parallel_class):
    def run_compile(fnlist):
        ths = [parallel_class(target=chooser, args=(fnlist,))
               for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    return run_compile


# workers
_threadsafe_backends = config.THREADING_LAYER in ('omppool', 'tbbpool')

_thread_class = threading.Thread


class _proc_class_impl(object):

    def __init__(self, method):
        self._method = method

    def __call__(self, *args, **kwargs):
        ctx = multiprocessing.get_context(self._method)
        return ctx.Process(*args, **kwargs)


thread_impl = compile_factory(_thread_class)
spawn_proc_impl = compile_factory(_proc_class_impl('spawn'))
fork_proc_impl = compile_factory(_proc_class_impl('fork'))


@unittest.skipUnless(_threadsafe_backends, "Threading layer not threadsafe")
class TestParallelBackend(unittest.TestCase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend
    """
    np.random.seed(42)

    all_impls = [jit_runner(nopython=True),
                 jit_runner(nopython=True, cache=True),
                 jit_runner(nopython=True, nogil=True),
                 jit_runner(nopython=True, parallel=True),
                 linalg_runner(nopython=True),
                 linalg_runner(nopython=True, nogil=True),
                 linalg_runner(nopython=True, parallel=True),
                 vectorize_runner(nopython=True),
                 vectorize_runner(nopython=True, target='parallel'),
                 guvectorize_runner(nopython=True),
                 guvectorize_runner(nopython=True, target='parallel'),
                 ]

    def run_compile(self, fnlist, parallelism='threading'):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            if parallelism == 'threading':
                thread_impl(fnlist)
            elif parallelism == 'multiprocessing_fork':
                fork_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_spawn':
                spawn_proc_impl(fnlist)
            elif parallelism == 'random':
                ps = [thread_impl, fork_proc_impl, spawn_proc_impl]
                for _ in range(10):  # 10 is arbitrary
                    impl = random.choice(ps)
                    impl(fnlist)
            else:
                raise ValueError(
                    'Unknown parallelism supplied %s' % parallelism)

    def test_threading_concurrent_vectorize(self):
        self.run_compile([vectorize_runner(nopython=True, target='parallel')],
                         parallelism='threading')

    def test_threading_concurrent_jit(self):
        self.run_compile([jit_runner(nopython=True, parallel=True)],
                         parallelism='threading')

    def test_threading_concurrent_guvectorize(self):
        self.run_compile([guvectorize_runner(nopython=True,
                                             target='parallel')],
                         parallelism='threading')

    def test_threading_concurrent_mix_use(self):
        self.run_compile(self.all_impls, parallelism='threading')

    def test_multiprocessing_fork_concurrent_vectorize(self):
        self.run_compile([vectorize_runner(nopython=True, target='parallel')],
                         parallelism='multiprocessing_fork')

    def test_multiprocessing_fork_concurrent_jit(self):
        self.run_compile([jit_runner(nopython=True, parallel=True)],
                         parallelism='multiprocessing_fork')

    def test_multiprocessing_fork_concurrent_guvectorize(self):
        self.run_compile([guvectorize_runner(nopython=True,
                                             target='parallel')],
                         parallelism='multiprocessing_fork')

    def test_multiprocessing_fork_concurrent_mix_use(self):
        self.run_compile(self.all_impls, parallelism='multiprocessing_fork')

    def test_multiprocessing_spawn_concurrent_vectorize(self):
        self.run_compile([vectorize_runner(nopython=True, target='parallel')],
                         parallelism='multiprocessing_spawn')

    def test_multiprocessing_spawn_concurrent_jit(self):
        self.run_compile([jit_runner(nopython=True, parallel=True)],
                         parallelism='multiprocessing_spawn')

    def test_multiprocessing_spawn_concurrent_guvectorize(self):
        self.run_compile([guvectorize_runner(nopython=True,
                                             target='parallel')],
                         parallelism='multiprocessing_spawn')

    def test_multiprocessing_fork_concurrent_mix_use(self):
        self.run_compile(self.all_impls, parallelism='multiprocessing_spawn')

    def test_random_concurrent_vectorize(self):
        self.run_compile([vectorize_runner(nopython=True, target='parallel')],
                         parallelism='random')

    def test_random_concurrent_jit(self):
        self.run_compile([jit_runner(nopython=True, parallel=True)],
                         parallelism='random')

    def test_random_concurrent_guvectorize(self):
        self.run_compile([guvectorize_runner(nopython=True,
                                             target='parallel')],
                         parallelism='random')

    def test_random_concurrent_mix_use(self):
        for _ in range(5):
            self.run_compile(self.all_impls, parallelism='random')


if __name__ == '__main__':
    unittest.main()
