"""Isolated control-case repro from tryforkissue.py — no fork.

Demonstrates: same Python process, writer creates a cached @njit wrapper
that gets serialized to disk, then a fresh wrapper sharing the same
SharedCacheLocator key loads the writer's .o into the SAME engine.
The engine already holds the strong-linkage `cfunc.<mangled>` symbol
from the writer, so `add_object_file` -> ORC DuplicateDefinition.

Expected (post-fix): no crash. Reader loads the cached overload and
reuses the already-published definitions.
"""
import os
import shutil
import tempfile

import numba
import numpy as np
from numba.core.caching import CacheImpl, _CacheLocator

CACHE_DIR = tempfile.mkdtemp(prefix="numba_fork_bug_control_")
FUNC_KEYS: dict = {}


class SharedCacheLocator(_CacheLocator):
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
    inner = numba.njit(impl_factory())
    name = inner.__name__
    src = f"def {name}(*args): return jitable_func(*args)"
    env = dict(globals())
    env["jitable_func"] = inner
    exec(compile(src, "<string>", "exec"), env)
    wrapper = env[name]
    FUNC_KEYS[wrapper] = cache_key
    return numba.njit(cache=True)(wrapper)


def returns_3d():
    def fn(x):
        return np.ascontiguousarray(x.transpose(2, 0, 1))
    return fn


if __name__ == "__main__":
    print(f"numba {numba.__version__}\n")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)

    # writer: compile + cache fn
    f_w = make_cached_op(returns_3d, "op_key")

    @numba.njit
    def run_w(x):
        return f_w(x)

    run_w(np.zeros((2, 3, 4)))
    print("writer: ok")

    # reader: fresh wrapper, same SharedCacheLocator key -> loads writer's .o
    # into the SAME engine where writer already published the symbols.
    f_r = make_cached_op(returns_3d, "op_key")

    @numba.njit
    def run_r(x):
        return f_r(x)

    try:
        run_r(np.zeros((2, 3, 4)))
        print("reader: ok")
    except Exception as e:
        print(f"reader: FAIL {type(e).__name__}: {e}")
        raise

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
