from __future__ import print_function, division, absolute_import

import numpy as np

import numba
from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dpctl.ocldrv as ocldrv
import sys
import io


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPLFallback(DPPLTestCase):

    def capture_stderr(self, func):
        backup = sys.stderr
        sys.stderr = io.StringIO()
        result = func()
        out = sys.stderr.getvalue()
        sys.stderr.close()
        sys.stderr = backup

        return out, result

    def test_dppl_fallback(self):

        @numba.jit
        def fill_value(i):
            return i

        def np_rand_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        def run_dppl():
            dppl = numba.njit(parallel={'offload':True})(np_rand_fallback)
            return dppl()

        ref = np_rand_fallback

        err, dppl_result = self.capture_stderr(run_dppl)
        ref_result = ref()

        np.testing.assert_array_equal(dppl_result, ref_result)
        self.assertTrue('Failed to lower parfor on GPU' in err)


if __name__ == '__main__':
    unittest.main()
