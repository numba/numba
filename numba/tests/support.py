"""
Assorted utilities for use in tests.
"""
from __future__ import print_function

import cmath
import contextlib
import enum
import errno
import gc
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import io
import ctypes
import multiprocessing as mp
import warnings
from contextlib import contextmanager

import numpy as np
try:
    import scipy
except ImportError:
    scipy = None

from numba import config, errors, typing, utils, numpy_support, testing
from numba.compiler import compile_extra, compile_isolated, Flags, DEFAULT_FLAGS
from numba.targets import cpu
import numba.unittest_support as unittest
from numba.runtime import rtsys
from numba.six import PY2


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()

nrt_flags = Flags()
nrt_flags.set("nrt")


tag = testing.make_tag_decorator(['important', 'long_running'])

_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_parfors_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)
skip_py38_or_later = unittest.skipIf(
    utils.PYVERSION >= (3, 8),
    "unsupported on py3.8 or later"
)
skip_tryexcept_unsupported = unittest.skipIf(
    utils.PYVERSION < (3, 7),
    "try-except unsupported on py3.6 or earlier"
)
skip_tryexcept_supported = unittest.skipIf(
    utils.PYVERSION >= (3, 7),
    "try-except supported on py3.7 or later"
)

_msg = "SciPy needed for test"
skip_unless_scipy = unittest.skipIf(scipy is None, _msg)


class CompilationCache(object):
    """
    A cache of compilation results for various signatures and flags.
    This can make tests significantly faster (or less slow).
    """

    def __init__(self):
        self.typingctx = typing.Context()
        self.targetctx = cpu.CPUContext(self.typingctx)
        self.cr_cache = {}

    def compile(self, func, args, return_type=None, flags=DEFAULT_FLAGS):
        """
        Compile the function or retrieve an already compiled result
        from the cache.
        """
        from numba.targets.registry import cpu_target

        cache_key = (func, args, return_type, flags)
        try:
            cr = self.cr_cache[cache_key]
        except KeyError:
            # Register the contexts in case for nested @jit or @overload calls
            # (same as compile_isolated())
            with cpu_target.nested_context(self.typingctx, self.targetctx):
                cr = compile_extra(self.typingctx, self.targetctx, func,
                                   args, return_type, flags, locals={})
            self.cr_cache[cache_key] = cr
        return cr


class TestCase(unittest.TestCase):

    longMessage = True

    # A random state yielding the same random numbers for any test case.
    # Use as `self.random.<method name>`
    @utils.cached_property
    def random(self):
        return np.random.RandomState(42)

    def reset_module_warnings(self, module):
        """
        Reset the warnings registry of a module.  This can be necessary
        as the warnings module is buggy in that regard.
        See http://bugs.python.org/issue4180
        """
        if isinstance(module, str):
            module = sys.modules[module]
        try:
            del module.__warningregistry__
        except AttributeError:
            pass

    @contextlib.contextmanager
    def assertTypingError(self):
        """
        A context manager that asserts the enclosed code block fails
        compiling in nopython mode.
        """
        _accepted_errors = (errors.LoweringError, errors.TypingError,
                            TypeError, NotImplementedError)
        with self.assertRaises(_accepted_errors) as cm:
            yield cm

    @contextlib.contextmanager
    def assertRefCount(self, *objects):
        """
        A context manager that asserts the given objects have the
        same reference counts before and after executing the
        enclosed block.
        """
        old_refcounts = [sys.getrefcount(x) for x in objects]
        yield
        new_refcounts = [sys.getrefcount(x) for x in objects]
        for old, new, obj in zip(old_refcounts, new_refcounts, objects):
            if old != new:
                self.fail("Refcount changed from %d to %d for object: %r"
                          % (old, new, obj))

    @contextlib.contextmanager
    def assertNoNRTLeak(self):
        """
        A context manager that asserts no NRT leak was created during
        the execution of the enclosed block.
        """
        old = rtsys.get_allocation_stats()
        yield
        new = rtsys.get_allocation_stats()
        total_alloc = new.alloc - old.alloc
        total_free = new.free - old.free
        total_mi_alloc = new.mi_alloc - old.mi_alloc
        total_mi_free = new.mi_free - old.mi_free
        self.assertEqual(total_alloc, total_free,
                         "number of data allocs != number of data frees")
        self.assertEqual(total_mi_alloc, total_mi_free,
                         "number of meminfo allocs != number of meminfo frees")


    _bool_types = (bool, np.bool_)
    _exact_typesets = [_bool_types, utils.INT_TYPES, (str,), (np.integer,),
                       (utils.text_type), (bytes, np.bytes_)]
    _approx_typesets = [(float,), (complex,), (np.inexact)]
    _sequence_typesets = [(tuple, list)]
    _float_types = (float, np.floating)
    _complex_types = (complex, np.complexfloating)

    def _detect_family(self, numeric_object):
        """
        This function returns a string description of the type family
        that the object in question belongs to.  Possible return values
        are: "exact", "complex", "approximate", "sequence", and "unknown"
        """
        if isinstance(numeric_object, np.ndarray):
            return "ndarray"

        if isinstance(numeric_object, enum.Enum):
            return "enum"

        for tp in self._sequence_typesets:
            if isinstance(numeric_object, tp):
                return "sequence"

        for tp in self._exact_typesets:
            if isinstance(numeric_object, tp):
                return "exact"

        for tp in self._complex_types:
            if isinstance(numeric_object, tp):
                return "complex"

        for tp in self._approx_typesets:
            if isinstance(numeric_object, tp):
                return "approximate"

        return "unknown"

    def _fix_dtype(self, dtype):
        """
        Fix the given *dtype* for comparison.
        """
        # Under 64-bit Windows, Numpy may return either int32 or int64
        # arrays depending on the function.
        if (sys.platform == 'win32' and sys.maxsize > 2**32 and
            dtype == np.dtype('int32')):
            return np.dtype('int64')
        else:
            return dtype

    def _fix_strides(self, arr):
        """
        Return the strides of the given array, fixed for comparison.
        Strides for 0- or 1-sized dimensions are ignored.
        """
        if arr.size == 0:
            return [0] * arr.ndim
        else:
            return [stride / arr.itemsize
                    for (stride, shape) in zip(arr.strides, arr.shape)
                    if shape > 1]

    def assertStridesEqual(self, first, second):
        """
        Test that two arrays have the same shape and strides.
        """
        self.assertEqual(first.shape, second.shape, "shapes differ")
        self.assertEqual(first.itemsize, second.itemsize, "itemsizes differ")
        self.assertEqual(self._fix_strides(first), self._fix_strides(second),
                         "strides differ")

    def assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                           msg=None, ignore_sign_on_zero=False,
                           abs_tol=None
                           ):
        """
        Versatile equality testing function with more built-in checks than
        standard assertEqual().

        For arrays, test that layout, dtype, shape are identical, and
        recursively call assertPreciseEqual() on the contents.

        For other sequences, recursively call assertPreciseEqual() on
        the contents.

        For scalars, test that two scalars or have similar types and are
        equal up to a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.
        ignore_sign_on_zero can be set to True if zeros are to be considered
        equal regardless of their sign bit.
        abs_tol if this is set to a float value its value is used in the
        following. If, however, this is set to the string "eps" then machine
        precision of the type(first) is used in the following instead. This
        kwarg is used to check if the absolute difference in value between first
        and second is less than the value set, if so the numbers being compared
        are considered equal. (This is to handle small numbers typically of
        magnitude less than machine precision).

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
        try:
            self._assertPreciseEqual(first, second, prec, ulps, msg,
                ignore_sign_on_zero, abs_tol)
        except AssertionError as exc:
            failure_msg = str(exc)
            # Fall off of the 'except' scope to avoid Python 3 exception
            # chaining.
        else:
            return
        # Decorate the failure message with more information
        self.fail("when comparing %s and %s: %s" % (first, second, failure_msg))

    def _assertPreciseEqual(self, first, second, prec='exact', ulps=1,
                            msg=None, ignore_sign_on_zero=False,
                            abs_tol=None):
        """Recursive workhorse for assertPreciseEqual()."""

        def _assertNumberEqual(first, second, delta=None):
            if (delta is None or first == second == 0.0
                or math.isinf(first) or math.isinf(second)):
                self.assertEqual(first, second, msg=msg)
                # For signed zeros
                if not ignore_sign_on_zero:
                    try:
                        if math.copysign(1, first) != math.copysign(1, second):
                            self.fail(
                                self._formatMessage(msg,
                                                    "%s != %s" %
                                                    (first, second)))
                    except TypeError:
                        pass
            else:
                self.assertAlmostEqual(first, second, delta=delta, msg=msg)

        first_family = self._detect_family(first)
        second_family = self._detect_family(second)

        assertion_message = "Type Family mismatch. (%s != %s)" % (first_family,
            second_family)
        if msg:
            assertion_message += ': %s' % (msg,)
        self.assertEqual(first_family, second_family, msg=assertion_message)

        # We now know they are in the same comparison family
        compare_family = first_family

        # For recognized sequences, recurse
        if compare_family == "ndarray":
            dtype = self._fix_dtype(first.dtype)
            self.assertEqual(dtype, self._fix_dtype(second.dtype))
            self.assertEqual(first.ndim, second.ndim,
                             "different number of dimensions")
            self.assertEqual(first.shape, second.shape,
                             "different shapes")
            self.assertEqual(first.flags.writeable, second.flags.writeable,
                             "different mutability")
            # itemsize is already checked by the dtype test above
            self.assertEqual(self._fix_strides(first),
                self._fix_strides(second), "different strides")
            if first.dtype != dtype:
                first = first.astype(dtype)
            if second.dtype != dtype:
                second = second.astype(dtype)
            for a, b in zip(first.flat, second.flat):
                self._assertPreciseEqual(a, b, prec, ulps, msg,
                                         ignore_sign_on_zero, abs_tol)
            return

        elif compare_family == "sequence":
            self.assertEqual(len(first), len(second), msg=msg)
            for a, b in zip(first, second):
                self._assertPreciseEqual(a, b, prec, ulps, msg,
                                         ignore_sign_on_zero, abs_tol)
            return

        elif compare_family == "exact":
            exact_comparison = True

        elif compare_family in ["complex", "approximate"]:
            exact_comparison = False

        elif compare_family == "enum":
            self.assertIs(first.__class__, second.__class__)
            self._assertPreciseEqual(first.value, second.value,
                                     prec, ulps, msg,
                                     ignore_sign_on_zero, abs_tol)
            return

        elif compare_family == "unknown":
            # Assume these are non-numeric types: we will fall back
            # on regular unittest comparison.
            self.assertIs(first.__class__, second.__class__)
            exact_comparison = True

        else:
            assert 0, "unexpected family"

        # If a Numpy scalar, check the dtype is exactly the same too
        # (required for datetime64 and timedelta64).
        if hasattr(first, 'dtype') and hasattr(second, 'dtype'):
            self.assertEqual(first.dtype, second.dtype)

        # Mixing bools and non-bools should always fail
        if (isinstance(first, self._bool_types) !=
            isinstance(second, self._bool_types)):
            assertion_message = ("Mismatching return types (%s vs. %s)"
                                 % (first.__class__, second.__class__))
            if msg:
                assertion_message += ': %s' % (msg,)
            self.fail(assertion_message)

        try:
            if cmath.isnan(first) and cmath.isnan(second):
                # The NaNs will compare unequal, skip regular comparison
                return
        except TypeError:
            # Not floats.
            pass

        # if absolute comparison is set, use it
        if abs_tol is not None:
            if abs_tol == "eps":
                rtol = np.finfo(type(first)).eps
            elif isinstance(abs_tol, float):
                rtol = abs_tol
            else:
                raise ValueError("abs_tol is not \"eps\" or a float, found %s"
                    % abs_tol)
            if abs(first - second) < rtol:
                return

        exact_comparison = exact_comparison or prec == 'exact'

        if not exact_comparison and prec != 'exact':
            if prec == 'single':
                bits = 24
            elif prec == 'double':
                bits = 53
            else:
                raise ValueError("unsupported precision %r" % (prec,))
            k = 2 ** (ulps - bits - 1)
            delta = k * (abs(first) + abs(second))
        else:
            delta = None
        if isinstance(first, self._complex_types):
            _assertNumberEqual(first.real, second.real, delta)
            _assertNumberEqual(first.imag, second.imag, delta)
        elif isinstance(first, (np.timedelta64, np.datetime64)):
            # Since Np 1.16 NaT == NaT is False, so special comparison needed
            if numpy_support.version >= (1, 16) and np.isnat(first):
                self.assertEqual(np.isnat(first), np.isnat(second))
            else:
                _assertNumberEqual(first, second, delta)
        else:
            _assertNumberEqual(first, second, delta)

    def run_nullary_func(self, pyfunc, flags):
        """
        Compile the 0-argument *pyfunc* with the given *flags*, and check
        it returns the same result as the pure Python function.
        The got and expected results are returned.
        """
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(got, expected)
        return got, expected

    if PY2:
        @contextmanager
        def subTest(self, *args, **kwargs):
            """A stub TestCase.subTest backport.
            This implementation is a no-op.
            """
            yield


class SerialMixin(object):
    """Mixin to mark test for serial execution.
    """
    _numba_parallel_test_ = False


# Various helpers

@contextlib.contextmanager
def override_config(name, value):
    """
    Return a context manager that temporarily sets Numba config variable
    *name* to *value*.  *name* must be the name of an existing variable
    in numba.config.
    """
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)


@contextlib.contextmanager
def override_env_config(name, value):
    """
    Return a context manager that temporarily sets an Numba config environment
    *name* to *value*.
    """
    old = os.environ.get(name)
    os.environ[name] = value
    config.reload_config()

    try:
        yield
    finally:
        if old is None:
            # If it wasn't set originally, delete the environ var
            del os.environ[name]
        else:
            # Otherwise, restore to the old value
            os.environ[name] = old
        # Always reload config
        config.reload_config()


def compile_function(name, code, globs):
    """
    Given a *code* string, compile it with globals *globs* and return
    the function named *name*.
    """
    co = compile(code.rstrip(), "<string>", "single")
    ns = {}
    eval(co, globs, ns)
    return ns[name]

def tweak_code(func, codestring=None, consts=None):
    """
    Tweak the code object of the given function by replacing its
    *codestring* (a bytes object) and *consts* tuple, optionally.
    """
    co = func.__code__
    tp = type(co)
    if codestring is None:
        codestring = co.co_code
    if consts is None:
        consts = co.co_consts
    if sys.version_info >= (3, 8):
        new_code = tp(co.co_argcount, co.co_posonlyargcount,
                      co.co_kwonlyargcount, co.co_nlocals,
                      co.co_stacksize, co.co_flags, codestring,
                      consts, co.co_names, co.co_varnames,
                      co.co_filename, co.co_name, co.co_firstlineno,
                      co.co_lnotab)
    elif sys.version_info >= (3,):
        new_code = tp(co.co_argcount, co.co_kwonlyargcount, co.co_nlocals,
                      co.co_stacksize, co.co_flags, codestring,
                      consts, co.co_names, co.co_varnames,
                      co.co_filename, co.co_name, co.co_firstlineno,
                      co.co_lnotab)
    else:
        new_code = tp(co.co_argcount, co.co_nlocals,
                      co.co_stacksize, co.co_flags, codestring,
                      consts, co.co_names, co.co_varnames,
                      co.co_filename, co.co_name, co.co_firstlineno,
                      co.co_lnotab)
    func.__code__ = new_code


_trashcan_dir = 'numba-tests'

if os.name == 'nt':
    # Under Windows, gettempdir() points to the user-local temp dir
    _trashcan_dir = os.path.join(tempfile.gettempdir(), _trashcan_dir)
else:
    # Mix the UID into the directory name to allow different users to
    # run the test suite without permission errors (issue #1586)
    _trashcan_dir = os.path.join(tempfile.gettempdir(),
                                 "%s.%s" % (_trashcan_dir, os.getuid()))

# Stale temporary directories are deleted after they are older than this value.
# The test suite probably won't ever take longer than this...
_trashcan_timeout = 24 * 3600  # 1 day

def _create_trashcan_dir():
    try:
        os.mkdir(_trashcan_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _purge_trashcan_dir():
    freshness_threshold = time.time() - _trashcan_timeout
    for fn in sorted(os.listdir(_trashcan_dir)):
        fn = os.path.join(_trashcan_dir, fn)
        try:
            st = os.stat(fn)
            if st.st_mtime < freshness_threshold:
                shutil.rmtree(fn, ignore_errors=True)
        except OSError as e:
            # In parallel testing, several processes can attempt to
            # remove the same entry at once, ignore.
            pass

def _create_trashcan_subdir(prefix):
    _purge_trashcan_dir()
    path = tempfile.mkdtemp(prefix=prefix + '-', dir=_trashcan_dir)
    return path

def temp_directory(prefix):
    """
    Create a temporary directory with the given *prefix* that will survive
    at least as long as this process invocation.  The temporary directory
    will be eventually deleted when it becomes stale enough.

    This is necessary because a DLL file can't be deleted while in use
    under Windows.

    An interesting side-effect is to be able to inspect the test files
    shortly after a test suite run.
    """
    _create_trashcan_dir()
    return _create_trashcan_subdir(prefix)


def import_dynamic(modname):
    """
    Import and return a module of the given name.  Care is taken to
    avoid issues due to Python's internal directory caching.
    """
    if sys.version_info >= (3, 3):
        import importlib
        importlib.invalidate_caches()
    __import__(modname)
    return sys.modules[modname]


# From CPython

@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, utils.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)

def captured_stdout():
    """Capture the output of sys.stdout:

       with captured_stdout() as stdout:
           print("hello")
       self.assertEqual(stdout.getvalue(), "hello\n")
    """
    return captured_output("stdout")

def captured_stderr():
    """Capture the output of sys.stderr:

       with captured_stderr() as stderr:
           print("hello", file=sys.stderr)
       self.assertEqual(stderr.getvalue(), "hello\n")
    """
    return captured_output("stderr")


@contextlib.contextmanager
def capture_cache_log():
    with captured_stdout() as out:
        with override_config('DEBUG_CACHE', True):
            yield out


class MemoryLeak(object):

    __enable_leak_check = True

    def memory_leak_setup(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        self.__init_stats = rtsys.get_allocation_stats()

    def memory_leak_teardown(self):
        if self.__enable_leak_check:
            self.assert_no_memory_leak()

    def assert_no_memory_leak(self):
        old = self.__init_stats
        new = rtsys.get_allocation_stats()
        total_alloc = new.alloc - old.alloc
        total_free = new.free - old.free
        total_mi_alloc = new.mi_alloc - old.mi_alloc
        total_mi_free = new.mi_free - old.mi_free
        self.assertEqual(total_alloc, total_free)
        self.assertEqual(total_mi_alloc, total_mi_free)

    def disable_leak_check(self):
        # For per-test use when MemoryLeakMixin is injected into a TestCase
        self.__enable_leak_check = False


class MemoryLeakMixin(MemoryLeak):

    def setUp(self):
        super(MemoryLeakMixin, self).setUp()
        self.memory_leak_setup()

    def tearDown(self):
        super(MemoryLeakMixin, self).tearDown()
        gc.collect()
        self.memory_leak_teardown()


@contextlib.contextmanager
def forbid_codegen():
    """
    Forbid LLVM code generation during the execution of the context
    manager's enclosed block.

    If code generation is invoked, a RuntimeError is raised.
    """
    from numba.targets import codegen
    patchpoints = ['CodeLibrary._finalize_final_module']

    old = {}
    def fail(*args, **kwargs):
        raise RuntimeError("codegen forbidden by test case")
    try:
        # XXX use the mock library instead?
        for name in patchpoints:
            parts = name.split('.')
            obj = codegen
            for attrname in parts[:-1]:
                obj = getattr(obj, attrname)
            attrname = parts[-1]
            value = getattr(obj, attrname)
            assert callable(value), ("%r should be callable" % name)
            old[obj, attrname] = value
            setattr(obj, attrname, fail)
        yield
    finally:
        for (obj, attrname), value in old.items():
            setattr(obj, attrname, value)


# For details about redirection of file-descriptor, read
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

@contextlib.contextmanager
def redirect_fd(fd):
    """
    Temporarily redirect *fd* to a pipe's write end and return a file object
    wrapping the pipe's read end.
    """

    from numba import _helperlib
    libnumba = ctypes.CDLL(_helperlib.__file__)

    libnumba._numba_flush_stdout()
    save = os.dup(fd)
    r, w = os.pipe()
    try:
        os.dup2(w, fd)
        yield io.open(r, "r")
    finally:
        libnumba._numba_flush_stdout()
        os.close(w)
        os.dup2(save, fd)
        os.close(save)


def redirect_c_stdout():
    """Redirect C stdout
    """
    fd = sys.__stdout__.fileno()
    return redirect_fd(fd)


def run_in_new_process_caching(func, cache_dir_prefix=__name__, verbose=True):
    """Spawn a new process to run `func` with a temporary cache directory.

    The childprocess's stdout and stderr will be captured and redirected to
    the current process's stdout and stderr.

    Returns
    -------
    ret : dict
        exitcode: 0 for success. 1 for exception-raised.
        stdout: str
        stderr: str
    """
    ctx = mp.get_context('spawn')
    qout = ctx.Queue()
    cache_dir = temp_directory(cache_dir_prefix)
    with override_env_config('NUMBA_CACHE_DIR', cache_dir):
        proc = ctx.Process(target=_remote_runner, args=[func, qout])
        proc.start()
        proc.join()
        stdout = qout.get_nowait()
        stderr = qout.get_nowait()
        if verbose and stdout.strip():
            print()
            print('STDOUT'.center(80, '-'))
            print(stdout)
        if verbose and stderr.strip():
            print(file=sys.stderr)
            print('STDERR'.center(80, '-'), file=sys.stderr)
            print(stderr, file=sys.stderr)
    return {
        'exitcode': proc.exitcode,
        'stdout': stdout,
        'stderr': stderr,
    }


def _remote_runner(fn, qout):
    """Used by `run_in_new_process_caching()`
    """
    with captured_stderr() as stderr:
        with captured_stdout() as stdout:
            try:
                fn()
            except Exception:
                traceback.print_exc()
                exitcode = 1
            else:
                exitcode = 0
        qout.put(stdout.getvalue())
    qout.put(stderr.getvalue())
    sys.exit(exitcode)

class CheckWarningsMixin(object):
    @contextlib.contextmanager
    def check_warnings(self, messages, category=RuntimeWarning):
        with warnings.catch_warnings(record=True) as catch:
            warnings.simplefilter("always")
            yield
        found = 0
        for w in catch:
            for m in messages:
                if m in str(w.message):
                    self.assertEqual(w.category, category)
                    found += 1
        self.assertEqual(found, len(messages))
