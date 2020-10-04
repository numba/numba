import numpy as np
import numba
from numba import dppl, njit, prange
from numba.dppl.testing import unittest, DPPLTestCase
from numba.tests.support import captured_stdout
import dpctl


def prange_example():
    n = 10
    a = np.ones((n, n), dtype=np.float64)
    b = np.ones((n, n), dtype=np.float64)
    c = np.ones((n, n), dtype=np.float64)
    for i in prange(n):
        a[i] = b[i] + c[i]


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestParforMessage(DPPLTestCase):
    def test_parfor_message(self):
        numba.dppl.compiler.DEBUG = 1
        jitted = njit(parallel={'offload':True})(prange_example)

        with captured_stdout() as got:
            with dpctl.device_context(dpctl.device_type.gpu, 0) as gpu_queue:
                jitted()

        self.assertTrue('Parfor lowered on DPPL-device' in got.getvalue())


if __name__ == '__main__':
    unittest.main()
