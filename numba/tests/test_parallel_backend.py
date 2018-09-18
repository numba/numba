# -*- coding: utf8 -*-
from __future__ import print_function, absolute_import

"""
Tests the parallel backend
"""
import threading
import multiprocessing
import random
import os
import sys
import subprocess
import signal

from numba import config, utils
if utils.PYVERSION >= (3, 0):
    import queue as t_queue
    import faulthandler
else:
    import Queue as t_queue

import numpy as np

from numba import unittest_support as unittest
from numba import jit, vectorize, guvectorize, njit

from .support import temp_directory, override_config, TestCase, tag

from .test_parfors import skip_unsupported as parfors_skip_unsupported
from .test_parfors import linux_only

# Check which backends are available
# TODO: Put this in a subprocess so the address space is kept clean
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

try:
    import scipy.linalg.cython_lapack
    _HAVE_LAPACK = True
except ImportError:
    _HAVE_LAPACK = False

# test skipping decorators
skip_no_omp = unittest.skipUnless(_HAVE_OMP_POOL, "OpenMP threadpool required")
skip_no_tbb = unittest.skipUnless(_HAVE_TBB_POOL, "TBB threadpool required")

_gnuomp = _HAVE_OMP_POOL and omppool.openmp_vendor == "GNU"
skip_unless_gnu_omp = unittest.skipUnless(_gnuomp, "GNU OpenMP only tests")

skip_unless_py3 = unittest.skipUnless(utils.PYVERSION >= (3, 0),
                                      "Test runs on Python 3 only")

_windows = sys.platform.startswith('win')
_osx = sys.platform.startswith('darwin')
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_parfors_unsupported = _32bit or _windows_py27

_HAVE_OS_FORK = not _windows


# some functions to jit

def foo(n, v):
    return np.ones(n) + v


if _HAVE_LAPACK:
    def linalg(n, v):
        x = np.dot(np.ones((n, n)), np.ones((n, n)))
        return x + np.arange(n) + v
else:
    def linalg(n, v):
        # no way to trigger MKL without the lapack bindings.
        return np.arange(n) + v


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
        np.testing.assert_allclose(expected, got)


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


def chooser(fnlist, **kwargs):
    q = kwargs.get('queue')
    try:
        if utils.PYVERSION >= (3, 0):
            faulthandler.enable()
        for _ in range(10):
            fn = random.choice(fnlist)
            fn()
    except BaseException as e:
        q.put(e)


def compile_factory(parallel_class, queue_impl):
    def run_compile(fnlist):
        q = queue_impl()
        kws = {'queue': q}
        ths = [parallel_class(target=chooser, args=(fnlist,), kwargs=kws)
               for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if not q.empty():
            errors = []
            while not q.empty():
                errors.append(q.get(False))
            _msg = "Error(s) occurred in delegated runner:\n%s"
            raise RuntimeError(_msg % '\n'.join([repr(x) for x in errors]))
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


def _get_mp_classes(method):
    if utils.PYVERSION < (3, 0):
        proc = _proc_class_impl(method)
        queue = multiprocessing.Queue
    else:
        if method == 'default':
            method = None
        ctx = multiprocessing.get_context(method)
        proc = _proc_class_impl(method)
        queue = ctx.Queue
    return proc, queue


thread_impl = compile_factory(_thread_class, t_queue.Queue)
spawn_proc_impl = compile_factory(*_get_mp_classes('spawn'))
if not _windows:
    fork_proc_impl = compile_factory(*_get_mp_classes('fork'))
    forkserver_proc_impl = compile_factory(*_get_mp_classes('forkserver'))

# this is duplication as Py27, linux uses fork, windows uses spawn, it however
# is kept like this so that when tests fail it's less confusing!
default_proc_impl = compile_factory(*_get_mp_classes('default'))


class TestParallelBackendBase(TestCase):
    """
    Base class for testing the parallel backends
    """

    all_impls = [jit_runner(nopython=True),
                 jit_runner(nopython=True, cache=True),
                 jit_runner(nopython=True, nogil=True),
                 linalg_runner(nopython=True),
                 linalg_runner(nopython=True, nogil=True),
                 vectorize_runner(nopython=True),
                 vectorize_runner(nopython=True, target='parallel'),
                 guvectorize_runner(nopython=True),
                 guvectorize_runner(nopython=True, target='parallel'),
                 ]

    if not _parfors_unsupported:
        parfor_impls = [jit_runner(nopython=True, parallel=True),
                        linalg_runner(nopython=True, parallel=True), ]
        all_impls.extend(parfor_impls)

    parallelism = ['threading', 'random']
    if utils.PYVERSION > (3, 0):
        parallelism.append('multiprocessing_spawn')
        if _HAVE_OS_FORK:
            parallelism.append('multiprocessing_fork')
            parallelism.append('multiprocessing_forkserver')
    else:
        parallelism.append('multiprocessing_default')

    runners = {'concurrent_jit': [jit_runner(nopython=True,
                                             parallel=(not _parfors_unsupported)
                                             )],
               'concurrect_vectorize':
                   [vectorize_runner(nopython=True, target='parallel')],
               'concurrent_guvectorize':
                   [guvectorize_runner(nopython=True, target='parallel')],
               'concurrent_mix_use': all_impls}

    safe_backends = {'omp', 'tbb'}

    def run_compile(self, fnlist, parallelism='threading'):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            if parallelism == 'threading':
                thread_impl(fnlist)
            elif parallelism == 'multiprocessing_fork':
                fork_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_forkserver':
                forkserver_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_spawn':
                spawn_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_default':
                default_proc_impl(fnlist)
            elif parallelism == 'random':
                if utils.PYVERSION < (3, 0):
                    ps = [thread_impl, default_proc_impl]
                else:
                    ps = [thread_impl, spawn_proc_impl]
                    if _HAVE_OS_FORK:
                        ps.append(fork_proc_impl)
                        ps.append(forkserver_proc_impl)

                random.shuffle(ps)
                for impl in ps:
                    impl(fnlist)
            else:
                raise ValueError(
                    'Unknown parallelism supplied %s' % parallelism)


_specific_backends = config.THREADING_LAYER in ('omp', 'tbb', 'workqueue')


@unittest.skipUnless(_specific_backends, "Threading layer not explicit")
class TestParallelBackend(TestParallelBackendBase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend.
    If a suitable backend is supplied via NUMBA_THREADING_LAYER these tests
    can be run directly. This test class cannot be run using the multiprocessing
    option to the test runner (i.e. `./runtests -m`) as daemon processes cannot
    have children.
    """

    # NOTE: All tests are generated based on what a platform supports concurrent
    # execution wise from Python, irrespective of whether the native libraries
    # can actually handle the behaviour present.
    @classmethod
    def generate(cls):
        for p in cls.parallelism:
            for name, impl in cls.runners.items():
                methname = "test_" + p + '_' + name

                def methgen(impl, p):
                    def test_method(self):
                        selfproc = multiprocessing.current_process()
                        # daemonized processes cannot have children
                        if selfproc.daemon:
                            _msg = 'daemonized processes cannot have children'
                            self.skipTest(_msg)
                        else:
                            self.run_compile(impl, parallelism=p)
                    return test_method
                setattr(cls, methname, methgen(impl, p))


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
    _DEBUG = False

    backends = {'tbb': skip_no_tbb,
                'omp': skip_no_omp,
                'workqueue': unittest.skipIf(False, '')}

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
                raise AssertionError(
                    "process failed with code %s: stderr follows\n%s\n" %
                    (popen.returncode, err.decode()))
            return out.decode(), err.decode()
        finally:
            timeout.cancel()
        return None, None

    def run_test_in_separate_process(self, test, threading_layer):
        env_copy = os.environ.copy()
        env_copy['NUMBA_THREADING_LAYER'] = str(threading_layer)
        cmdline = [sys.executable, "-m", "numba.runtests", test]
        return self.run_cmd(cmdline, env_copy)

    @classmethod
    def _inject(cls, p, name, backend, backend_guard):
        themod = cls.__module__
        thecls = TestParallelBackend.__name__
        methname = "test_" + p + '_' + name
        injected_method = '%s.%s.%s' % (themod, thecls, methname)

        def test_template(self):
            o, e = self.run_test_in_separate_process(injected_method, backend)
            if self._DEBUG:
                print('stdout:\n "%s"\n stderr:\n "%s"' % (o, e))
            self.assertIn('OK', e)
            self.assertTrue('FAIL' not in e)
            self.assertTrue('ERROR' not in e)
        injected_test = "test_%s_%s_%s" % (p, name, backend)
        # Mark as important for appveyor
        setattr(cls, injected_test,
                tag("important")(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        for backend, backend_guard in cls.backends.items():
            for p in cls.parallelism:
                for name in cls.runners.keys():
                    # handle known problem cases...

                    # GNU OpenMP is not fork safe
                    if (p in ('multiprocessing_fork', 'random') and
                        backend == 'omp' and
                            sys.platform.startswith('linux')):
                        continue

                    # workqueue is not thread safe
                    if (p in ('threading', 'random') and
                            backend == 'workqueue'):
                        continue

                    cls._inject(p, name, backend, backend_guard)


TestSpecificBackend.generate()


class ThreadLayerTestHelper(TestCase):
    """
    Helper class for running an isolated piece of code based on a template
    """
    # sys path injection and separate usecase module to make sure everything
    # is importable by children of multiprocessing
    _here = "%r" % os.path.dirname(__file__)

    template = """if 1:
    import sys
    sys.path.insert(0, "%(here)r")
    import multiprocessing
    import numpy as np
    from numba import njit
    import numba
    try:
        import threading_backend_usecases
    except ImportError as e:
        print("DEBUG:", sys.path)
        raise e
    import os

    sigterm_handler = threading_backend_usecases.sigterm_handler
    busy_func = threading_backend_usecases.busy_func

    def the_test():
        %%s

    if __name__ == "__main__":
        the_test()
    """ % {'here': _here}

    def run_cmd(self, cmdline, env=None):
        if env is None:
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str("omp")
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
                raise AssertionError(
                    "process failed with code %s: stderr follows\n%s\n" %
                    (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return out.decode(), err.decode()


@parfors_skip_unsupported
class TestThreadingLayerSelection(ThreadLayerTestHelper):
    """
    Checks that numba.threading_layer() reports correctly.
    """
    _DEBUG = False

    backends = {'tbb': skip_no_tbb,
                'omp': skip_no_omp,
                'workqueue': unittest.skipIf(False, '')}

    @classmethod
    def _inject(cls, backend, backend_guard):

        def test_template(self):
            body = """if 1:
                X = np.arange(1000000.)
                Y = np.arange(1000000.)
                Z = busy_func(X, Y)
                assert numba.threading_layer() == '%s'
            """
            runme = self.template % (body % backend)
            cmdline = [sys.executable, '-c', runme]
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str(backend)
            out, err = self.run_cmd(cmdline, env=env)
            if self._DEBUG:
                print(out, err)
        injected_test = "test_threading_layer_selector_%s" % backend
        setattr(cls, injected_test,
                tag("important")(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        for backend, backend_guard in cls.backends.items():
            cls._inject(backend, backend_guard)


TestThreadingLayerSelection.generate()


# 32bit or windows py27 (not that this runs on windows)
@parfors_skip_unsupported
@skip_unless_gnu_omp
class TestForkSafetyIssues(ThreadLayerTestHelper):
    """
    Checks Numba's behaviour in various situations involving GNU OpenMP and fork
    """
    _DEBUG = False

    def test_check_threading_layer_is_gnu(self):
        runme = """if 1:
            from numba.npyufunc import omppool
            assert omppool.openmp_vendor == 'GNU'
            """
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)

    def test_par_parent_os_fork_par_child(self):
        """
        Whilst normally valid, this actually isn't for Numba invariant of OpenMP
        Checks SIGABRT is received.
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            Z = busy_func(X, Y)
            pid = os.fork()
            if pid  == 0:
                Z = busy_func(X, Y)
            else:
                os.wait()
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        try:
            out, err = self.run_cmd(cmdline)
        except AssertionError as e:
            self.assertIn("failed with code -6", str(e))

    def test_par_parent_implicit_mp_fork_par_child(self):
        """
        Implicit use of multiprocessing fork context.
        Does this:
        1. Start with OpenMP
        2. Fork to processes using OpenMP (this is invalid)
        3. Joins fork
        4. Check the exception pushed onto the queue that is a result of
           catching SIGTERM coming from the C++ aborting on illegal fork
           pattern for GNU OpenMP
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = multiprocessing.Queue()

            # Start OpenMP runtime on parent via parallel function
            Z = busy_func(X, Y, q)

            # fork() underneath with no exec, will abort
            proc = multiprocessing.Process(target = busy_func, args=(X, Y, q))
            proc.start()

            err = q.get()
            assert "Caught SIGTERM" in str(err)
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @linux_only
    @skip_unless_py3
    def test_par_parent_explicit_mp_fork_par_child(self):
        """
        Explicit use of multiprocessing fork context.
        Does this:
        1. Start with OpenMP
        2. Fork to processes using OpenMP (this is invalid)
        3. Joins fork
        4. Check the exception pushed onto the queue that is a result of
           catching SIGTERM coming from the C++ aborting on illegal fork
           pattern for GNU OpenMP
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = multiprocessing.Queue()

            # Start OpenMP runtime on parent via parallel function
            Z = busy_func(X, Y, q)

            # fork() underneath with no exec, will abort
            ctx = multiprocessing.get_context('fork')
            proc = ctx.Process(target = busy_func, args=(X, Y, q))
            proc.start()
            proc.join()

            err = q.get()
            assert "Caught SIGTERM" in str(err)
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @skip_unless_py3
    def test_par_parent_mp_spawn_par_child_par_parent(self):
        """
        Explicit use of multiprocessing spawn, this is safe.
        Does this:
        1. Start with OpenMP
        2. Spawn to processes using OpenMP
        3. Join spawns
        4. Run some more OpenMP
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = multiprocessing.Queue()

            # Start OpenMP runtime and run on parent via parallel function
            Z = busy_func(X, Y, q)
            procs = []
            ctx = multiprocessing.get_context('spawn')
            for x in range(20): # start a lot to try and get overlap
                ## fork() + exec() to run some OpenMP on children
                proc = ctx.Process(target = busy_func, args=(X, Y, q))
                procs.append(proc)
                sys.stdout.flush()
                sys.stderr.flush()
                proc.start()

            [p.join() for p in procs]

            try:
                q.get(False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise RuntimeError("Queue was not empty")

            # Run some more OpenMP on parent
            Z = busy_func(X, Y, q)
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    def test_serial_parent_implicit_mp_fork_par_child_then_par_parent(self):
        """
        Implicit use of multiprocessing (will be fork, but cannot declare that
        in Py2.7 as there's no process launch context).
        Does this:
        1. Start with no OpenMP
        2. Fork to processes using OpenMP
        3. Join forks
        4. Run some OpenMP
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = multiprocessing.Queue()

            # this is ok
            procs = []
            for x in range(10):
                # fork() underneath with but no OpenMP in parent, this is ok
                proc = multiprocessing.Process(target = busy_func,
                                               args=(X, Y, q))
                procs.append(proc)
                proc.start()

            [p.join() for p in procs]

            # and this is still ok as the OpenMP happened in forks
            Z = busy_func(X, Y, q)
            try:
                q.get(False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise RuntimeError("Queue was not empty")
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @linux_only
    @skip_unless_py3
    def test_serial_parent_explicit_mp_fork_par_child_then_par_parent(self):
        """
        Explicit use of multiprocessing 'fork'.
        Does this:
        1. Start with no OpenMP
        2. Fork to processes using OpenMP
        3. Join forks
        4. Run some OpenMP
        """
        body = """if 1:
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = multiprocessing.Queue()

            # this is ok
            procs = []
            ctx = multiprocessing.get_context('fork')
            for x in range(10):
                # fork() underneath with but no OpenMP in parent, this is ok
                proc = ctx.Process(target = busy_func, args=(X, Y, q))
                procs.append(proc)
                proc.start()

            [p.join() for p in procs]

            # and this is still ok as the OpenMP happened in forks
            Z = busy_func(X, Y, q)
            try:
                q.get(False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise RuntimeError("Queue was not empty")
        """
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)


if __name__ == '__main__':
    unittest.main()
