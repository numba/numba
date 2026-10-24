import os
import sys
import unittest
from unittest import mock

from numba.core import codegen
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, override_config, run_in_subprocess


class TestAccelerateLoading(TestCase):
    def setUp(self):
        global_compiler_lock.acquire()
        self.addCleanup(global_compiler_lock.release)
        codegen._load_accelerate.cache_clear()
        self.addCleanup(codegen._load_accelerate.cache_clear)

    def test_load_once(self):
        with override_config('VECTOR_MATH_LIBRARY', 'accelerate'), \
             mock.patch.object(codegen, 'find_library',
                               return_value='libAccelerate') as find, \
             mock.patch.object(codegen.ll, 'load_library_permanently') as load:
            codegen.JITCPUCodegen('first')
            codegen.JITCPUCodegen('second')
        find.assert_called_once_with('Accelerate')
        load.assert_called_once_with('libAccelerate')

    def test_missing_library(self):
        with mock.patch.object(codegen, 'find_library',
                               side_effect=[None, 'libAccelerate']) as find, \
             mock.patch.object(codegen.ll, 'load_library_permanently') as load:
            with self.assertRaisesRegex(RuntimeError,
                                        'Accelerate.*could not be found'):
                codegen._load_accelerate()
            load.assert_not_called()
            codegen._load_accelerate()
        self.assertEqual(find.call_count, 2)
        load.assert_called_once_with('libAccelerate')

    def test_load_failure(self):
        with mock.patch.object(codegen, 'find_library',
                               return_value='libAccelerate') as find, \
             mock.patch.object(codegen.ll, 'load_library_permanently',
                               side_effect=[RuntimeError('cannot load'),
                                            None]) as load:
            with self.assertRaisesRegex(RuntimeError, 'cannot load'):
                codegen._load_accelerate()
            codegen._load_accelerate()
        self.assertEqual(find.call_count, 2)
        self.assertEqual(load.call_args_list,
                         [mock.call('libAccelerate')] * 2)

    def test_no_library(self):
        with mock.patch.object(codegen, 'find_library') as find, \
             mock.patch.object(codegen.ll, 'load_library_permanently') as load:
            for provider in (None, 'none'):
                with self.subTest(provider=provider), \
                     override_config('VECTOR_MATH_LIBRARY', provider):
                    codegen.JITCPUCodegen('no_vector_library')
        find.assert_not_called()
        load.assert_not_called()

    def test_aot(self):
        with override_config('VECTOR_MATH_LIBRARY', 'accelerate'), \
             mock.patch.object(codegen, 'find_library') as find, \
             mock.patch.object(codegen.ll, 'load_library_permanently') as load:
            codegen.AOTCPUCodegen('aot')
        find.assert_not_called()
        load.assert_not_called()

    @unittest.skipUnless(sys.platform == 'darwin', 'requires Accelerate')
    def test_sine(self):
        code = """if 1:
            import numpy as np
            from numba import config, jit

            config.VECTOR_MATH_LIBRARY = 'accelerate'

            @jit
            def sine(x):
                y = np.empty_like(x)
                for i in range(x.size):
                    y[i] = np.sin(x[i])
                return y

            x = np.linspace(-2, 2, 128, dtype=np.float32)
            np.testing.assert_allclose(sine(x), np.sin(x),
                                       rtol=2e-6, atol=2e-7)
            llvm_ir = sine.inspect_llvm(sine.signatures[0])
            assert any('call ' in line and '@vsinf(' in line
                       for line in llvm_ir.splitlines()), llvm_ir
        """
        env = os.environ.copy()
        env.pop('NUMBA_VECTOR_MATH_LIBRARY', None)
        run_in_subprocess(code, env=env)


if __name__ == '__main__':
    unittest.main()
