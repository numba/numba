from __future__ import print_function, absolute_import

import os
import platform
import textwrap

from .support import TestCase, override_config, captured_stdout, forbid_codegen
from numba import unittest_support as unittest
from numba import jit, jitclass, types
from numba.compiler import compile_isolated


def simple_nopython(somearg):
    retval = somearg + 1
    return retval

def simple_gen(x, y):
    yield x
    yield y


class SimpleClass(object):
    def __init__(self):
        self.h = 5

simple_class_spec = [('h', types.int32)]

def simple_class_user(obj):
    return obj.h


class DebugTestBase(TestCase):

    all_dumps = set(['bytecode', 'cfg', 'ir', 'typeinfer', 'llvm',
                     'func_opt_llvm', 'optimized_llvm', 'assembly'])

    def assert_fails(self, *args, **kwargs):
        self.assertRaises(AssertionError, *args, **kwargs)

    def check_debug_output(self, out, dump_names):
        enabled_dumps = dict.fromkeys(self.all_dumps, False)
        for name in dump_names:
            assert name in enabled_dumps
            enabled_dumps[name] = True
        for name, enabled in sorted(enabled_dumps.items()):
            check_meth = getattr(self, '_check_dump_%s' % name)
            if enabled:
                check_meth(out)
            else:
                self.assert_fails(check_meth, out)

    def _check_dump_bytecode(self, out):
        self.assertIn('BINARY_ADD', out)

    def _check_dump_cfg(self, out):
        self.assertIn('CFG dominators', out)

    def _check_dump_ir(self, out):
        self.assertIn('--IR DUMP: %s--' % self.func_name, out)

    def _check_dump_typeinfer(self, out):
        self.assertIn('--propagate--', out)

    def _check_dump_llvm(self, out):
        self.assertIn('--LLVM DUMP', out)
        self.assertIn('%"retval" = alloca', out)

    def _check_dump_func_opt_llvm(self, out):
        self.assertIn('--FUNCTION OPTIMIZED DUMP %s' % self.func_name, out)
        # allocas have been optimized away
        self.assertIn('add nsw i64 %arg.somearg, 1', out)

    def _check_dump_optimized_llvm(self, out):
        self.assertIn('--OPTIMIZED DUMP %s' % self.func_name, out)
        self.assertIn('add nsw i64 %arg.somearg, 1', out)

    def _check_dump_assembly(self, out):
        self.assertIn('--ASSEMBLY %s' % self.func_name, out)
        if platform.machine() in ('x86_64', 'AMD64', 'i386', 'i686'):
            self.assertIn('xorl', out)


class FunctionDebugTestBase(DebugTestBase):

    func_name = 'simple_nopython'

    def compile_simple_nopython(self):
        with captured_stdout() as out:
            cres = compile_isolated(simple_nopython, (types.int64,))
            # Sanity check compiled function
            self.assertPreciseEqual(cres.entry_point(2), 3)
        return out.getvalue()


class TestFunctionDebugOutput(FunctionDebugTestBase):

    def test_dump_bytecode(self):
        with override_config('DUMP_BYTECODE', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['bytecode'])

    def test_dump_ir(self):
        with override_config('DUMP_IR', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['ir'])

    def test_dump_cfg(self):
        with override_config('DUMP_CFG', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['cfg'])

    def test_dump_llvm(self):
        with override_config('DUMP_LLVM', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['llvm'])

    def test_dump_func_opt_llvm(self):
        with override_config('DUMP_FUNC_OPT', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['func_opt_llvm'])

    def test_dump_optimized_llvm(self):
        with override_config('DUMP_OPTIMIZED', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['optimized_llvm'])

    def test_dump_assembly(self):
        with override_config('DUMP_ASSEMBLY', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['assembly'])


class TestGeneratorDebugOutput(DebugTestBase):

    func_name = 'simple_gen'

    def compile_simple_gen(self):
        with captured_stdout() as out:
            cres = compile_isolated(simple_gen, (types.int64, types.int64))
            # Sanity check compiled function
            self.assertPreciseEqual(list(cres.entry_point(2, 5)), [2, 5])
        return out.getvalue()

    def test_dump_ir_generator(self):
        with override_config('DUMP_IR', True):
            out = self.compile_simple_gen()
        self.check_debug_output(out, ['ir'])
        self.assertIn('--GENERATOR INFO: %s' % self.func_name, out)
        expected_gen_info = textwrap.dedent("""
            generator state variables: ['x', 'y']
            yield point #1: live variables = ['y'], weak live variables = ['x']
            yield point #2: live variables = [], weak live variables = ['y']
            """)
        self.assertIn(expected_gen_info, out)


class TestDisableJIT(DebugTestBase):
    """
    Test the NUMBA_DISABLE_JIT environment variable.
    """

    def test_jit(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                cfunc = jit(nopython=True)(simple_nopython)
                self.assertPreciseEqual(cfunc(2), 3)

    def test_jitclass(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                SimpleJITClass = jitclass(simple_class_spec)(SimpleClass)

                obj = SimpleJITClass()
                self.assertPreciseEqual(obj.h, 5)

                cfunc = jit(nopython=True)(simple_class_user)
                self.assertPreciseEqual(cfunc(obj), 5)


class TestEnvironmentOverride(FunctionDebugTestBase):
    """
    Test that environment variables are reloaded by Numba when modified.
    """

    def test_debug(self):
        out = self.compile_simple_nopython()
        self.assertFalse(out)
        os.environ['NUMBA_DEBUG'] = '1'
        try:
            out = self.compile_simple_nopython()
            # Note that all variables dependent on NUMBA_DEBUG are
            # updated too.
            self.check_debug_output(out, ['ir', 'typeinfer',
                                          'llvm', 'func_opt_llvm',
                                          'optimized_llvm', 'assembly'])
        finally:
            del os.environ['NUMBA_DEBUG']
        out = self.compile_simple_nopython()
        self.assertFalse(out)


if __name__ == '__main__':
    unittest.main()
