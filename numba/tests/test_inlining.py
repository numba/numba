from __future__ import print_function, absolute_import

from .support import TestCase, override_config, captured_stdout
from numba import unittest_support as unittest
from numba import jit, types
from numba.compiler import compile_isolated


@jit((types.int32,), nopython=True)
def inner(a):
    return a + 1

@jit((types.int32,), nopython=True)
def more(a):
    return inner(inner(a))

def outer_simple(a):
    return inner(a) * 2

def outer_multiple(a):
    return inner(a) * more(a)


class TestInlining(TestCase):
    """
    Check that jitted inner functions are inlined into outer functions,
    in nopython mode.
    Note that not all inner functions are guaranteed to be inlined.
    We just trust LLVM's inlining heuristics.
    """

    def test_inner_function(self):
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_simple)
        self.assertPreciseEqual(cfunc(1), 4)
        # Check the inner function was elided from the output (which also
        # guarantees it was inlined into the outer function).
        asm = out.getvalue()
        prefix = __name__
        self.assertIn('%s.outer_simple' % prefix, asm)
        self.assertNotIn('%s.inner' % prefix, asm)

    def test_multiple_inner_functions(self):
        # Same with multiple inner functions, and multiple calls to
        # the same inner function (inner()).  This checks that linking in
        # the same library/module twice doesn't produce linker errors.
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_multiple)
        self.assertPreciseEqual(cfunc(1), 6)
        asm = out.getvalue()
        prefix = __name__
        self.assertIn('%s.outer_multiple' % prefix, asm)
        self.assertNotIn('%s.more' % prefix, asm)
        self.assertNotIn('%s.inner' % prefix, asm)


if __name__ == '__main__':
    unittest.main()
