"""Numba bug: LLVM type mismatch when loading fork-cached code.

Root cause
----------
numba.core.bytecode.FunctionIdentity._unique_ids is a process-global counter
(itertools.count). After os.fork(), parent and child independently assign the
SAME UIDs to DIFFERENT functions. UIDs are embedded in LLVM mangled symbol
names. When the parent loads a fork-child's cached compilation and also freshly
compiles a different function that happens to get the same (qualname, uid,
argtypes), LLVM sees two definitions of the same symbol with different return
types -> ValueError.

Required ingredients
--------------------
1. Custom _CacheLocator (directs multiple exec()-created functions to a shared
   cache directory, keyed by a content-based disambiguator)
2. exec()-created wrapper functions (all wrappers share the same qualname even
   though they wrap different implementations)
3. os.fork() (child inherits the UID counter -> counter overlap with parent)
4. At the colliding UID offset, the fork-cached function and the parent's
   freshly compiled function must differ in return type (e.g. 3D vs 4D array)

Impact on PyTensor: the numba backend creates per-op wrapper functions via
exec(), all DimShuffle ops share the qualname "dimshuffle", and ASV uses
os.fork() for benchmarks. When fork child 1 caches DimShuffle(2,0,1) (3D->3D)
and fork child 2 compiles DimShuffle('x',0,1,2) (3D->4D) at the same UID
offset, the type mismatch crashes the benchmark.
"""
import os
import shutil
import tempfile

import numba
import numpy as np
from numba.core.caching import CacheImpl, _CacheLocator

CACHE_DIR = tempfile.mkdtemp(prefix="numba_fork_bug_")
FUNC_KEYS: dict = {}


class SharedCacheLocator(_CacheLocator):
    """Direct all registered functions to a shared cache directory."""

    def __init__(self, py_func, py_file, key):
        self._py_func, self._py_file, self._key = py_func, py_file, key

    def ensure_cache_path(self):
        pass

    def get_cache_path(self):
        return CACHE_DIR

    def get_source_stamp(self):
        return 0

    def get_disambiguator(self):
        return self._key

    @classmethod
    def from_function(cls, py_func, py_file):
        if py_func in FUNC_KEYS:
            return cls(py_func, py_file, FUNC_KEYS[py_func])
        return None


CacheImpl._locator_classes.insert(0, SharedCacheLocator)


def make_cached_op(impl_factory, cache_key):
    """Create a cached @njit wrapper via exec().

    Each call creates a fresh inner @njit function from impl_factory(),
    then wraps it in an exec()-created function. The exec() wrapper has
    the same qualname as the inner function regardless of what the inner
    function actually does. This, combined with UID counter overlap after
    fork, creates the LLVM symbol collision.
    """
    inner = numba.njit(impl_factory())
    name = inner.__name__
    src = f"def {name}(*args): return jitable_func(*args)"
    env = dict(globals())
    env["jitable_func"] = inner
    exec(compile(src, "<string>", "exec"), env)
    wrapper = env[name]
    FUNC_KEYS[wrapper] = cache_key
    return numba.njit(cache=True)(wrapper)


# Two operations that return DIFFERENT types but share the function name "fn".
# This models PyTensor's DimShuffle: all DimShuffle ops produce functions
# named "dimshuffle" regardless of the specific axis permutation.

def returns_3d():
    """Like DimShuffle(2,0,1): 3D input -> 3D output."""
    def fn(x):
        return np.ascontiguousarray(x.transpose(2, 0, 1))
    return fn


def returns_4d():
    """Like DimShuffle('x',0,1,2): 3D input -> 4D output."""
    def fn(x):
        return x.reshape((1,) + x.shape)
    return fn


def run_in_fork(func):
    pid = os.fork()
    if pid == 0:
        try:
            func()
            os._exit(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            os._exit(1)
    else:
        _, status = os.waitpid(pid, 0)
        return os.WEXITSTATUS(status)


if __name__ == "__main__":
    print(f"numba {numba.__version__}\n")

    # ── BUG: fork child caches fn(3D->3D), parent compiles fn(3D->4D) ──
    # Both wrappers are named "fn" (from exec()). After fork, both get
    # the same UID -> same LLVM mangled name -> type mismatch.

    def writer():
        f = make_cached_op(returns_3d, "op_key")
        @numba.njit
        def run(x):
            return f(x)
        run(np.zeros((2, 3, 4)))

    def reader():
        f3 = make_cached_op(returns_3d, "op_key")    # loads fork's cache
        f4 = make_cached_op(returns_4d, "op_key_2")  # compiles fresh, SAME uid
        @numba.njit
        def run(x):
            return f3(x), f4(x)
        run(np.zeros((2, 3, 4)))

    print("Test: fork child writes cache -> parent reads")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    ec = run_in_fork(writer)
    print(f"  fork exit={ec}")
    try:
        reader()
        print("  -> OK (no crash)")
    except ValueError as e:
        print(f"  -> BUG: ValueError: {e}")

    print("\nControl: same process (no fork)")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    writer()
    try:
        reader()
        print("  -> OK")
    except ValueError as e:
        print(f"  -> BUG: ValueError: {e}")

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
