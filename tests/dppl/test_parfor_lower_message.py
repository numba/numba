import numpy as np
import numba
from numba import dppl, njit, prange
from numba.dppl.testing import unittest, DPPLTestCase
from numba.tests.support import captured_stdout
import dpctl.ocldrv as ocldrv


def prange_example():
    n = 10
    a = np.ones((n), dtype=np.float64)
    b = np.ones((n), dtype=np.float64)
    c = np.ones((n), dtype=np.float64)
    for i in prange(n//2):
        a[i] = b[i] + c[i]

    return a


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestParforMessage(DPPLTestCase):
    def test_parfor_message(self):
        numba.dppl.compiler.DEBUG = 1
        jitted = njit(parallel={'offload':True})(prange_example)

        with captured_stdout() as got:
            jitted()

        numba.dppl.compiler.DEBUG = 0
        self.assertTrue('Parfor lowered on DPPL-device' in got.getvalue())


if __name__ == '__main__':
    unittest.main()
