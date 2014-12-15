from __future__ import print_function, absolute_import

import platform

from .support import TestCase, override_config, captured_stdout
from numba import unittest_support as unittest
from numba import types
from numba.compiler import compile_isolated


def simple_nopython(somearg):
    retval = somearg + 1
    return retval


class TestDebugOutput(TestCase):

    def compile_simple_nopython(self):
        with captured_stdout() as out:
            cres = compile_isolated(simple_nopython, (types.int64,))
            # Sanity check compiled function
            self.assertPreciseEqual(cres.entry_point(2), 3)
        return out.getvalue()

    def assert_fails(self, *args, **kwargs):
        self.assertRaises(AssertionError, *args, **kwargs)

    def check_debug_output(self, out, enabled_dumps):
        all_dumps = dict.fromkeys(['bytecode', 'cfg', 'ir', 'llvm',
                                   'func_opt_llvm', 'optimized_llvm',
                                   'assembly'],
                                  False)
        for name in enabled_dumps:
            assert name in all_dumps
            all_dumps[name] = True
        for name, enabled in sorted(all_dumps.items()):
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
        self.assertIn('--IR DUMP: simple_nopython--', out)

    def _check_dump_llvm(self, out):
        self.assertIn('--LLVM DUMP', out)
        self.assertIn('%"retval" = alloca', out)

    def _check_dump_func_opt_llvm(self, out):
        self.assertIn('--FUNCTION OPTIMIZED DUMP simple_nopython', out)
        # allocas have been optimized away
        self.assertIn('add i64 %arg.somearg, 1', out)

    def _check_dump_optimized_llvm(self, out):
        self.assertIn('--OPTIMIZED DUMP simple_nopython', out)
        self.assertIn('add i64 %arg.somearg, 1', out)

    def _check_dump_assembly(self, out):
        self.assertIn('--ASSEMBLY simple_nopython', out)
        if platform.machine() in ('x86_64', 'AMD64', 'i386', 'i686'):
            self.assertIn('xorl', out)

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


if __name__ == '__main__':
    unittest.main()
