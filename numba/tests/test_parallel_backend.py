# -*- coding: utf-8 -*-

"""
Tests the parallel backend
"""
import faulthandler
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import unittest

import numpy as np

from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
                                 skip_parfors_unsupported, linux_only)

import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config


_TEST_TIMEOUT = _RUNNER_TIMEOUT - 60.


# Check which backends are available
# TODO: Put this in a subprocess so the address space is kept clean
try:
    # Check it's a compatible TBB before loading it
    from numba.np.ufunc.parallel import _check_tbb_version_compatible
    _check_tbb_version_compatible()
    from numba.np.ufunc import tbbpool    # noqa: F401
    _HAVE_TBB_POOL = True
except ImportError:
    _HAVE_TBB_POOL = False

try:
    from numba.np.ufunc import omppool
    _HAVE_OMP_POOL = True
except ImportError:
    _HAVE_OMP_POOL = False

try:
    import scipy.linalg.cython_lapack    # noqa: F401
    _HAVE_LAPACK = True
except ImportError:
    _HAVE_LAPACK = False

# test skipping decorators
skip_no_omp = unittest.skipUnless(_HAVE_OMP_POOL, "OpenMP threadpool required")
skip_no_tbb = unittest.skipUnless(_HAVE_TBB_POOL, "TBB threadpool required")

_gnuomp = _HAVE_OMP_POOL and omppool.openmp_vendor == "GNU"
skip_unless_gnu_omp = unittest.skipUnless(_gnuomp, "GNU OpenMP only tests")

_windows = sys.platform.startswith('win')
_osx = sys.platform.startswith('darwin')
_32bit = sys.maxsize <= 2 ** 32
_parfors_unsupported = _32bit

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


class mask_runner(object):
    def __init__(self, runner, mask, **options):
        self.runner = runner
        self.mask = mask

    def __call__(self):
        if self.mask:
            # Tests are all run in isolated subprocesses, so we
            # don't have to worry about this affecting other tests
            set_num_threads(self.mask)
        self.runner()


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
        faulthandler.enable()
        for _ in range(int(len(fnlist) * 1.5)):
            fn = random.choice(fnlist)
            fn()
    except Exception as e:
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
        ctx = multiprocessing.get_context(self._method)
        return ctx.Process(*args, **kwargs)


def _get_mp_classes(method):
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

    all_impls = [
        jit_runner(nopython=True),
        jit_runner(nopython=True, cache=True),
        jit_runner(nopython=True, nogil=True),
        linalg_runner(nopython=True),
        linalg_runner(nopython=True, nogil=True),
        vectorize_runner(nopython=True),
        vectorize_runner(nopython=True, target='parallel'),
        vectorize_runner(nopython=True, target='parallel', cache=True),
        guvectorize_runner(nopython=True),
        guvectorize_runner(nopython=True, target='parallel'),
        guvectorize_runner(nopython=True, target='parallel', cache=True),
    ]

    if not _parfors_unsupported:
        parfor_impls = [
            jit_runner(nopython=True, parallel=True),
            jit_runner(nopython=True, parallel=True, cache=True),
            linalg_runner(nopython=True, parallel=True),
            linalg_runner(nopython=True, parallel=True, cache=True),
        ]
        all_impls.extend(parfor_impls)

    if config.NUMBA_NUM_THREADS < 2:
        # Not enough cores
        masks = []
    else:
        masks = [1, 2]

    mask_impls = []
    for impl in all_impls:
        for mask in masks:
            mask_impls.append(mask_runner(impl, mask))

    parallelism = ['threading', 'random']
    parallelism.append('multiprocessing_spawn')
    if _HAVE_OS_FORK:
        parallelism.append('multiprocessing_fork')
        parallelism.append('multiprocessing_forkserver')

    runners = {
        'concurrent_jit': [
            jit_runner(nopython=True, parallel=(not _parfors_unsupported)),
        ],
        'concurrect_vectorize': [
            vectorize_runner(nopython=True, target='parallel'),
        ],
        'concurrent_guvectorize': [
            guvectorize_runner(nopython=True, target='parallel'),
        ],
        'concurrent_mix_use': all_impls,
        'concurrent_mix_use_masks': mask_impls,
    }

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
                fn = methgen(impl, p)
                fn.__name__ = methname
                setattr(cls, methname, fn)


TestParallelBackend.generate()


class TestInSubprocess(object):
    backends = {'tbb': skip_no_tbb,
                'omp': skip_no_omp,
                'workqueue': unittest.skipIf(False, '')}

    def run_cmd(self, cmdline, env):
        popen = subprocess.Popen(cmdline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env)
        # finish in _TEST_TIMEOUT seconds or kill it
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
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


class TestSpecificBackend(TestInSubprocess, TestParallelBackendBase):
    """
    This is quite contrived, for each test in the TestParallelBackend tests it
    generates a test that will run the TestParallelBackend test in a new python
    process with an environment modified to ensure a specific threadsafe backend
    is used. This is with view of testing the backends independently and in an
    isolated manner such that if they hang/crash/have issues, it doesn't kill
    the test suite.
    """
    _DEBUG = False

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
        # Mark as long_running
        setattr(cls, injected_test,
                tag('long_running')(backend_guard(test_template)))

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
        # finish in _TEST_TIMEOUT seconds or kill it
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
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


@skip_parfors_unsupported
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


@skip_parfors_unsupported
class TestMiscBackendIssues(ThreadLayerTestHelper):
    """
    Checks fixes for the issues with threading backends implementation
    """
    _DEBUG = False

    @skip_no_omp
    def test_omp_stack_overflow(self):
        """
        Tests that OMP does not overflow stack
        """
        runme = """if 1:
            from numba import vectorize, threading_layer
            import numpy as np

            @vectorize(['f4(f4,f4,f4,f4,f4,f4,f4,f4)'], target='parallel')
            def foo(a, b, c, d, e, f, g, h):
                return a+b+c+d+e+f+g+h

            x = np.ones(2**20, np.float32)
            foo(*([x]*8))
            print("@%s@" % threading_layer())
        """
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = "omp"
        env['OMP_STACKSIZE'] = "100K"
        out, err = self.run_cmd(cmdline, env=env)
        if self._DEBUG:
            print(out, err)
        self.assertIn("@omp@", out)

    @skip_no_tbb
    def test_single_thread_tbb(self):
        """
        Tests that TBB works well with single thread
        https://github.com/numba/numba/issues/3440
        """
        runme = """if 1:
            from numba import njit, prange, threading_layer

            @njit(parallel=True)
            def foo(n):
                acc = 0
                for i in prange(n):
                    acc += i
                return acc

            foo(100)
            print("@%s@" % threading_layer())
        """
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = "tbb"
        env['NUMBA_NUM_THREADS'] = "1"
        out, err = self.run_cmd(cmdline, env=env)
        if self._DEBUG:
            print(out, err)
        self.assertIn("@tbb@", out)

    def test_workqueue_aborts_on_nested_parallelism(self):
        """
        Tests workqueue raises sigabrt if a nested parallel call is performed
        """
        runme = """if 1:
            from numba import njit, prange
            import numpy as np

            @njit(parallel=True)
            def nested(x):
                for i in prange(len(x)):
                    x[i] += 1


            @njit(parallel=True)
            def main():
                Z = np.zeros((5, 10))
                for i in prange(Z.shape[0]):
                    nested(Z[i])
                return Z

            main()
        """
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = "workqueue"
        env['NUMBA_NUM_THREADS'] = "4"

        try:
            out, err = self.run_cmd(cmdline, env=env)
        except AssertionError as e:
            if self._DEBUG:
                print(out, err)
            e_msg = str(e)
            self.assertIn("failed with code", e_msg)
            # raised a SIGABRT, but the value is platform specific so just check
            # the error message
            self.assertIn("Terminating: Nested parallel kernel launch detected",
                          e_msg)


# 32bit or windows py27 (not that this runs on windows)
@skip_parfors_unsupported
@skip_unless_gnu_omp
class TestForkSafetyIssues(ThreadLayerTestHelper):
    """
    Checks Numba's behaviour in various situations involving GNU OpenMP and fork
    """
    _DEBUG = False

    def test_check_threading_layer_is_gnu(self):
        runme = """if 1:
            from numba.np.ufunc import omppool
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
            mp = multiprocessing.get_context('fork')
            X = np.arange(1000000.)
            Y = np.arange(1000000.)
            q = mp.Queue()

            # Start OpenMP runtime on parent via parallel function
            Z = busy_func(X, Y, q)

            # fork() underneath with no exec, will abort
            proc = mp.Process(target = busy_func, args=(X, Y, q))
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


@skip_parfors_unsupported
class TestInitSafetyIssues(TestCase):

    _DEBUG = False

    @linux_only # only linux can leak semaphores
    def test_orphaned_semaphore(self):
        # sys path injection and separate usecase module to make sure everything
        # is importable by children of multiprocessing

        def run_cmd(cmdline):
            popen = subprocess.Popen(cmdline,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,)
            # finish in _TEST_TIMEOUT seconds or kill it
            timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
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

        test_file = os.path.join(os.path.dirname(__file__),
                                 "orphaned_semaphore_usecase.py")

        cmdline = [sys.executable, test_file]
        out, err = run_cmd(cmdline)

        # assert no semaphore leaks reported on stderr
        self.assertNotIn("leaked semaphore", err)

        if self._DEBUG:
            print("OUT:", out)
            print("ERR:", err)


@skip_parfors_unsupported
@skip_no_omp
class TestOpenMPVendors(TestCase):

    def test_vendors(self):
        """
        Checks the OpenMP vendor strings are correct
        """
        expected = dict()
        expected['win32'] = "MS"
        expected['darwin'] = "Intel"
        expected['linux'] = "GNU"

        # only check OS that are supported, custom toolchains may well work as
        # may other OS
        for k in expected.keys():
            if sys.platform.startswith(k):
                self.assertEqual(expected[k], omppool.openmp_vendor)


if __name__ == '__main__':
    unittest.main()
