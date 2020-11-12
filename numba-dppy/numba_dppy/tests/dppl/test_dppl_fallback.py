from __future__ import print_function, division, absolute_import

import numpy as np

import numba
from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
from numba.tests.support import captured_stderr
import dpctl
import sys
import io


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPLFallback(DPPLTestCase):
    def test_dppl_fallback_inner_call(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        with captured_stderr() as msg:
            dppl = numba.njit(parallel={'offload':True})(inner_call_fallback)
            dppl_result = dppl()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(dppl_result, ref_result)
        self.assertTrue('Failed to lower parfor on DPPL-device' in msg.getvalue())

    def test_dppl_fallback_reductions(self):
        def reduction(a):
            return np.amax(a)

        a = np.ones(10)
        with captured_stderr() as msg:
            dppl = numba.njit(parallel={'offload':True})(reduction)
            dppl_result = dppl(a)

        ref_result = reduction(a)

        np.testing.assert_array_equal(dppl_result, ref_result)
        self.assertTrue('Failed to lower parfor on DPPL-device' in msg.getvalue())


if __name__ == '__main__':
    unittest.main()
