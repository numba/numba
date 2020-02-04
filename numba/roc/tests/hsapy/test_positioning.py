import numpy as np
from numba import roc
import unittest


class TestPositioning(unittest.TestCase):

    def test_kernel_jit(self):
        @roc.jit
        def udt(output):
            global_id = roc.get_global_id(0)
            global_size = roc.get_global_size(0)
            local_id = roc.get_local_id(0)
            group_id = roc.get_group_id(0)
            num_groups = roc.get_num_groups(0)
            workdim = roc.get_work_dim()
            local_size = roc.get_local_size(0)

            output[0, group_id, local_id] = global_id
            output[1, group_id, local_id] = global_size
            output[2, group_id, local_id] = local_id
            output[3, group_id, local_id] = local_size
            output[4, group_id, local_id] = group_id
            output[5, group_id, local_id] = num_groups
            output[6, group_id, local_id] = workdim

        out = np.zeros((7, 2, 3), dtype=np.intp)
        udt[2, 3](out)

        np.testing.assert_equal([[0, 1, 2], [3, 4, 5]], out[0])
        np.testing.assert_equal(6, out[1])
        np.testing.assert_equal([[0, 1, 2]] * 2, out[2])
        np.testing.assert_equal(3, out[3])
        np.testing.assert_equal([[0, 0, 0], [1, 1, 1]], out[4])
        np.testing.assert_equal(2, out[5])
        np.testing.assert_equal(1, out[6])


if __name__ == '__main__':
    unittest.main()

