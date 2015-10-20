from __future__ import print_function, absolute_import

import numpy as np
from numba import hsa
from numba.hsa.hsadrv.error import HsaDriverError
import numba.unittest_support as unittest


class TestSimple(unittest.TestCase):
    def test_array_access(self):
        magic_token = 123

        @hsa.jit
        def udt(output):
            output[0] = magic_token

        out = np.zeros(1, dtype=np.intp)
        udt[1, 1](out)

        self.assertEqual(out[0], magic_token)

    def test_array_access_2d(self):
        magic_token = 123

        @hsa.jit
        def udt(output):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    output[i, j] = magic_token

        out = np.zeros((10, 10), dtype=np.intp)
        udt[1, 1](out)
        np.testing.assert_equal(out, magic_token)

    def test_array_access_3d(self):
        magic_token = 123

        @hsa.jit
        def udt(output):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    for k in range(output.shape[2]):
                        output[i, j, k] = magic_token

        out = np.zeros((10, 10, 10), dtype=np.intp)
        udt[1, 1](out)
        np.testing.assert_equal(out, magic_token)

    def test_global_id(self):
        @hsa.jit
        def udt(output):
            global_id = hsa.get_global_id(0)
            output[global_id] = global_id

        # Allocate extra space to track bad indexing
        out = np.zeros(100 + 2, dtype=np.intp)
        udt[10, 10](out[1:-1])

        np.testing.assert_equal(out[1:-1], np.arange(100))

        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 0)

    def test_local_id(self):
        @hsa.jit
        def udt(output):
            global_id = hsa.get_global_id(0)
            local_id = hsa.get_local_id(0)
            output[global_id] = local_id

        # Allocate extra space to track bad indexing
        out = np.zeros(100 + 2, dtype=np.intp)
        udt[10, 10](out[1:-1])

        subarr = out[1:-1]

        for parted in np.split(subarr, 10):
            np.testing.assert_equal(parted, np.arange(10))

        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 0)

    def test_group_id(self):
        @hsa.jit
        def udt(output):
            global_id = hsa.get_global_id(0)
            group_id = hsa.get_group_id(0)
            output[global_id] = group_id + 1

        # Allocate extra space to track bad indexing
        out = np.zeros(100 + 2, dtype=np.intp)
        udt[10, 10](out[1:-1])

        subarr = out[1:-1]

        for i, parted in enumerate(np.split(subarr, 10), start=1):
            np.testing.assert_equal(parted, i)

        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 0)

    def test_workdim(self):
        @hsa.jit
        def udt(output):
            global_id = hsa.get_global_id(0)
            workdim = hsa.get_work_dim()
            output[global_id] = workdim

        out = np.zeros(10, dtype=np.intp)
        udt[1, 10](out)
        np.testing.assert_equal(out, 1)

        @hsa.jit
        def udt2(output):
            g0 = hsa.get_global_id(0)
            g1 = hsa.get_global_id(1)
            output[g0, g1] = hsa.get_work_dim()

        out = np.zeros((2, 5), dtype=np.intp)
        udt2[(1, 1), (2, 5)](out)
        np.testing.assert_equal(out, 2)

    def test_empty_kernel(self):
        @hsa.jit
        def udt():
            pass

        udt[1, 1]()

    def test_workgroup_oversize(self):
        @hsa.jit
        def udt():
            pass

        with self.assertRaises(HsaDriverError) as raises:
            udt[1, 2**30]()
        self.assertIn("workgroupsize is too big", str(raises.exception))


if __name__ == '__main__':
    unittest.main()
