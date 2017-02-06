from __future__ import print_function, absolute_import

import math
import numpy as np

from numba import unittest_support as unittest
from numba.tests.support import captured_stdout, override_config
from numba import njit, vectorize, guvectorize


class TestFastMath(unittest.TestCase):
    def test_jit(self):
        def foo(x):
            return x + math.sin(x)
        fastfoo = njit(fastmath=True)(foo)
        slowfoo = njit(foo)
        self.assertEqual(fastfoo(0.5), slowfoo(0.5))
        fastllvm = fastfoo.inspect_llvm(fastfoo.signatures[0])
        slowllvm = slowfoo.inspect_llvm(slowfoo.signatures[0])
        # Ensure fast attribute in fast version only
        self.assertIn('fadd fast', fastllvm)
        self.assertIn('call fast', fastllvm)
        self.assertNotIn('fadd fast', slowllvm)
        self.assertNotIn('call fast', slowllvm)

    def test_vectorize(self):
        def foo(x):
            return x + math.sin(x)
        fastfoo = vectorize(fastmath=True)(foo)
        slowfoo = vectorize(foo)
        x = np.random.random(8).astype(np.float32)
        # capture the optimized llvm to check for fast flag
        with override_config('DUMP_OPTIMIZED', True):
            with captured_stdout() as slow_cap:
                expect = slowfoo(x)
            slowllvm = slow_cap.getvalue()
            with captured_stdout() as fast_cap:
                got = fastfoo(x)
            fastllvm = fast_cap.getvalue()
        np.testing.assert_almost_equal(expect, got)
        self.assertIn('fadd fast', fastllvm)
        self.assertIn('call fast', fastllvm)
        self.assertNotIn('fadd fast', slowllvm)
        self.assertNotIn('call fast', slowllvm)

    def test_guvectorize(self):
        def foo(x, out):
            out[0] = x + math.sin(x)
        x = np.random.random(8).astype(np.float32)
        with override_config('DUMP_OPTIMIZED', True):
            types = ['(float32, float32[:])']
            sig = '()->()'
            with captured_stdout() as fast_cap:
                fastfoo = guvectorize(types, sig, fastmath=True)(foo)
            fastllvm = fast_cap.getvalue()
            with captured_stdout() as slow_cap:
                slowfoo = guvectorize(types, sig)(foo)
            slowllvm = slow_cap.getvalue()
        expect = slowfoo(x)
        got = fastfoo(x)
        np.testing.assert_almost_equal(expect, got)
        self.assertIn('fadd fast', fastllvm)
        self.assertIn('call fast', fastllvm)
        self.assertNotIn('fadd fast', slowllvm)
        self.assertNotIn('call fast', slowllvm)


if __name__ == '__main__':
    unittest.main()
