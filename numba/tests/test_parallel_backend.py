"""
Tests the parallel backend
"""
import threading
import multiprocessing
import random
import os
import sys
import subprocess

import numpy as np

from numba import config, utils
from numba import unittest_support as unittest
from numba import jit, vectorize, guvectorize

from .support import temp_directory, override_config, TestCase

# Check which backends are available.
try:
    from numba.npyufunc import tbbpool
    _HAVE_TBB_POOL = True
except ImportError:
    _HAVE_TBB_POOL = False

try:
    from numba.npyufunc import omppool
    _HAVE_OMP_POOL = True
except ImportError:
    _HAVE_OMP_POOL = False

skip_no_omp = unittest.skipUnless(_HAVE_OMP_POOL, "OpenMP threadpool required")
skip_no_tbb = unittest.skipUnless(_HAVE_TBB_POOL, "TBB threadpool required")


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
        sig = ['(f4, f4, f4[:])']
        cfunc = guvectorize(sig, '(),()->()', **self._options)(gufunc_foo)
        a = b = np.random.random(10).astype(np.float32)
        expected = ufunc_foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)

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
_thread_class = threading.Thread


class _proc_class_impl(object):

    def __init__(self, method):
        self._method = method

    def __call__(self, *args, **kwargs):
        if utils.PYVERSION < (3, 0):
            return multiprocessing.Process(*args, **kwargs)
        else:
            ctx = multiprocessing.get_context(self._method)
            return ctx.Process(*args, **kwargs)


thread_impl = compile_factory(_thread_class)
spawn_proc_impl = compile_factory(_proc_class_impl('spawn'))
fork_proc_impl = compile_factory(_proc_class_impl('fork'))

# this is duplication as Py27, linux uses fork, windows uses spawn, it however
# is kept like this so that when tests fail it's less confusing!
default_proc_impl = compile_factory(_proc_class_impl('default'))


class TestParallelBackendBase(TestCase):

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
    if utils.PYVERSION < (3, 0):
        parallelism = ['threading', 'multiprocessing_default', 'random']
    else:
        parallelism = ['threading', 'multiprocessing_fork',
                       'multiprocessing_spawn', 'random']

    runners = {'concurrent_jit': [jit_runner(nopython=True, parallel=True)],
               'concurrect_vectorize': [vectorize_runner(nopython=True, target='parallel')],
               'concurrent_guvectorize': [guvectorize_runner(nopython=True, target='parallel')],
               'concurrent_mix_use': all_impls}

    safe_backends = {'omppool', 'tbbpool'}

    def run_compile(self, fnlist, parallelism='threading'):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            if parallelism == 'threading':
                thread_impl(fnlist)
            elif parallelism == 'multiprocessing_fork':
                fork_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_spawn':
                spawn_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_default':
                default_proc_impl(fnlist)
            elif parallelism == 'random':
                if utils.PYVERSION < (3, 0):
                    ps = [thread_impl, default_proc_impl]
                else:
                    ps = [thread_impl, fork_proc_impl, spawn_proc_impl]

                for _ in range(10):  # 10 is arbitrary
                    impl = random.choice(ps)
                    impl(fnlist)
            else:
                raise ValueError(
                    'Unknown parallelism supplied %s' % parallelism)


_threadsafe_backends = config.THREADING_LAYER in ('omppool', 'tbbpool')


@unittest.skipUnless(_threadsafe_backends, "Threading layer not threadsafe")
class TestParallelBackend(TestParallelBackendBase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend.
    If a suitable backend is supplied via NUMBA_THREADING_LAYER these tests
    can be run directly.
    """

    @classmethod
    def generate(cls):
        for p in cls.parallelism:
            for name, impl in cls.runners.items():
                def test_method(self):
                    self.run_compile(impl, parallelism=p)
                methname = "test_" + p + '_' + name
                setattr(cls, methname, test_method)


TestParallelBackend.generate()


class TestSpecificBackend(TestParallelBackendBase):
    """
    This is quite contrived, for each test in the TestParallelBackend tests it
    generates a test that will run the TestParallelBackend test in a new python
    process with an environment modified to ensure a specific threadsafe backend
    is used. This is with view of testing the backends independently and in an
    isolated manner such that if they hang/crash/have issues, it doesn't kill
    the test suite.
    """

    backends = {'tbbpool': skip_no_tbb,
                'omppool': skip_no_omp}

    def run_cmd(self, cmdline, env):
        popen = subprocess.Popen(cmdline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env)
        # finish in 5 minutes or kill it
        timeout = threading.Timer(5 * 60., popen.kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError("process failed with code %s: stderr follows\n%s\n"
                                     % (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return out.decode(), err.decode()

    def run_test_in_separate_process(self, test, threading_layer):
        env_copy = os.environ.copy()
        env_copy['NUMBA_THREADING_LAYER'] = str(threading_layer)
        print("Running %s with backend: %s" % (test, threading_layer))
        cmdline = [sys.executable, "-m", "numba.runtests", test]
        return self.run_cmd(cmdline, env_copy)

    @classmethod
    def _inject(cls, p, name, backend, backend_guard):
        themod = cls.__module__
        thecls = TestParallelBackend.__name__
        methname = "test_" + p + '_' + name
        injected_method = '%s.%s.%s' % (themod, thecls, methname)

        def test_template(self):
            self.run_test_in_separate_process(injected_method, backend)
        injected_test = "test_%s_%s_%s" % (p, name, backend)
        setattr(cls, injected_test, backend_guard(test_template))

    @classmethod
    def generate(cls):
        for backend, backend_guard in cls.backends.items():
            for p in cls.parallelism:
                for name in cls.runners.keys():
                    cls._inject(p, name, backend, backend_guard)


TestSpecificBackend.generate()


if __name__ == '__main__':
    unittest.main()
