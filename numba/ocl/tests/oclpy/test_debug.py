from __future__ import print_function, absolute_import

import numpy as np

from numba.tests.support import override_config, captured_stderr, captured_stdout
from numba import unittest_support as unittest
from numba.ocl.testing import OCLTestCase
from numba import ocl, float64


def simple_ocl(A, B):
    i = ocl.get_global_id(0)
    B[i] = A[i] + 1.5


class TestDebugOutput(OCLTestCase):

    def compile_simple_ocl(self):
        with captured_stderr() as err:
            with captured_stdout() as out:
                cfunc = ocl.jit((float64[:], float64[:]))(simple_ocl)
                # Call compiled function to ensure code is generated
                # and sanity-check results.
                A = np.linspace(0, 1, 10).astype(np.float64)
                B = np.zeros_like(A)
                cfunc[1, 10](A, B)
                self.assertTrue(np.allclose(A + 1.5, B))
        # stderr shouldn't be affected by debug output
        self.assertFalse(err.getvalue())
        return out.getvalue()

    def assert_fails(self, *args, **kwargs):
        self.assertRaises(AssertionError, *args, **kwargs)

    def check_debug_output(self, out, enabled_dumps):
        all_dumps = dict.fromkeys(['bytecode', 'cfg', 'ir', 'llvm',
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
                self.assertRaises(AssertionError, check_meth, out)

    def _check_dump_bytecode(self, out):
        self.assertIn('BINARY_ADD', out)

    def _check_dump_cfg(self, out):
        self.assertIn('CFG dominators', out)

    def _check_dump_ir(self, out):
        self.assertIn('--IR DUMP: simple_ocl--', out)
        self.assertIn('const(float, 1.5)', out)

    def _check_dump_llvm(self, out):
        self.assertIn('--LLVM DUMP', out)

    def _check_dump_assembly(self, out):
        self.assertIn('--ASSEMBLY', out)
        self.assertIn('SPIR-V', out)

    def test_dump_bytecode(self):
        with override_config('DUMP_BYTECODE', True):
            out = self.compile_simple_ocl()
        self.check_debug_output(out, ['bytecode'])

    def test_dump_ir(self):
        with override_config('DUMP_IR', True):
            out = self.compile_simple_ocl()
        self.check_debug_output(out, ['ir'])

    def test_dump_cfg(self):
        with override_config('DUMP_CFG', True):
            out = self.compile_simple_ocl()
        self.check_debug_output(out, ['cfg'])

    def test_dump_llvm(self):
        with override_config('DUMP_LLVM', True):
            out = self.compile_simple_ocl()
        self.check_debug_output(out, ['llvm'])

    def test_dump_assembly(self):
        with override_config('DUMP_ASSEMBLY', True):
            out = self.compile_simple_ocl()
        self.check_debug_output(out, ['assembly'])


if __name__ == '__main__':
    unittest.main()
