from __future__ import print_function

import warnings

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba.utils import PYVERSION
from numba import types
from .support import TestCase, CompilationCache


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def assignments(a):
    b = c = str(a)
    return b + c


def assignments2(a):
    b = c = d = str(a)
    return b + c + d


# Use cases for issue #503

def var_propagate1(a, b):
    c = (a if a > b else b) + 5
    return c


def var_propagate2(a, b):
    c = 5 + (a if a > b else b + 12) / 2.0
    return c


def var_propagate3(a, b):
    c = 5 + (a > b and a or b)
    return c


def var_propagate4(a, b):
    c = 5 + (a - 1 and b + 1) or (a + 1 and b - 1)
    return c


# Issue #480
def chained_compare(a):
    return 1 < a < 3


# Issue #591
def stack_effect_error(x):
    i = 2
    c = 1
    if i == x:
        for i in range(3):
            c = i
    return i + c

# Some more issues with stack effect and blocks
def for_break(n, x):
    for i in range(n):
        n = 0
        if i == x:
            break
    else:
        n = i
    return i, n

# Issue #571
def var_swapping(a, b, c, d, e):
    a, b = b, a
    c, d, e = e, c, d
    a, b, c, d = b, c, d, a
    return a + b + c + d +e


class TestDataFlow(TestCase):

    def setUp(self):
        self.cache = CompilationCache()
        # All tests here should run without warnings
        self.w_cm = warnings.catch_warnings()
        self.w_cm.__enter__()
        warnings.simplefilter("error")

    def tearDown(self):
        self.w_cm.__exit__(None, None, None)

    def test_assignments(self, flags=force_pyobj_flags):
        pyfunc = assignments
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_assignments2(self, flags=force_pyobj_flags):
        pyfunc = assignments2
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

        if flags is force_pyobj_flags:
            cfunc("a")

    # The dataflow analysis must be good enough for native mode
    # compilation to succeed, hence the no_pyobj_flags in the following tests.

    def run_propagate_func(self, pyfunc, args):
        cr = self.cache.compile(pyfunc, (types.int32, types.int32),
                                flags=no_pyobj_flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))

    def test_var_propagate1(self):
        self.run_propagate_func(var_propagate1, (2, 3))
        self.run_propagate_func(var_propagate1, (3, 2))

    def test_var_propagate2(self):
        self.run_propagate_func(var_propagate2, (2, 3))
        self.run_propagate_func(var_propagate2, (3, 2))

    def test_var_propagate3(self):
        self.run_propagate_func(var_propagate3, (2, 3))
        self.run_propagate_func(var_propagate3, (3, 2))
        self.run_propagate_func(var_propagate3, (2, 0))
        self.run_propagate_func(var_propagate3, (-1, 0))
        self.run_propagate_func(var_propagate3, (0, 2))
        self.run_propagate_func(var_propagate3, (0, -1))

    def test_var_propagate4(self):
        self.run_propagate_func(var_propagate4, (1, 1))
        self.run_propagate_func(var_propagate4, (1, 0))
        self.run_propagate_func(var_propagate4, (1, -1))
        self.run_propagate_func(var_propagate4, (0, 1))
        self.run_propagate_func(var_propagate4, (0, 0))
        self.run_propagate_func(var_propagate4, (0, -1))
        self.run_propagate_func(var_propagate4, (-1, 1))
        self.run_propagate_func(var_propagate4, (-1, 0))
        self.run_propagate_func(var_propagate4, (-1, -1))

    def test_chained_compare(self, flags=force_pyobj_flags):
        pyfunc = chained_compare
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in [0, 1, 2, 3, 4]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_chained_compare_npm(self):
        self.test_chained_compare(no_pyobj_flags)

    def test_stack_effect_error(self, flags=force_pyobj_flags):
        # Issue #591: POP_BLOCK must undo all stack pushes done inside
        # the block.
        pyfunc = stack_effect_error
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cr.entry_point
        for x in (0, 1, 2, 3):
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_stack_effect_error_npm(self):
        self.test_stack_effect_error(no_pyobj_flags)

    def test_var_swapping(self, flags=force_pyobj_flags):
        pyfunc = var_swapping
        cr = compile_isolated(pyfunc, (types.int32,) * 5, flags=flags)
        cfunc = cr.entry_point
        args = tuple(range(0, 10, 2))
        self.assertPreciseEqual(pyfunc(*args), cfunc(*args))

    def test_var_swapping_npm(self):
        self.test_var_swapping(no_pyobj_flags)

    def test_for_break(self, flags=force_pyobj_flags):
        # BREAK_LOOP must unwind the current inner syntax block.
        pyfunc = for_break
        cr = compile_isolated(pyfunc, (types.intp, types.intp), flags=flags)
        cfunc = cr.entry_point
        for (n, x) in [(4, 2), (4, 6)]:
            self.assertPreciseEqual(pyfunc(n, x), cfunc(n, x))

    def test_for_break_npm(self):
        self.test_for_break(no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()

