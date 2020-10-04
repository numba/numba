from __future__ import print_function, division, absolute_import

import numpy as np

from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dpctl.ocldrv as ocldrv


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestPYDPPLAPI(DPPLTestCase):
    def test_pydppl_api(self):
        A = np.array(np.random.random(10), dtype=np.float32)
        with ocldrv.igpu_context(0) as device_env:
            # DeviceEnv APIs
            dA = device_env.copy_array_to_device(A)

            device_env.copy_array_from_device(dA)

            dB = device_env.create_device_array(A) # A is given here as meta-data

            device_env.dump()

            max_work_group_size = device_env.get_max_work_group_size()
            max_work_item_dims  = device_env.get_max_work_item_dims()


            # DeviceArray APIs
            self.assertTrue(hex(id(A)) == hex(id(dA.get_ndarray())))

if __name__ == '__main__':
    unittest.main()
