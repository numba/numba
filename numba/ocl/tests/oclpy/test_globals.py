from __future__ import absolute_import, print_function, division
import numpy as np
from numba import ocl, int32, float32
from numba.ocl.testing import unittest
from numba.ocl.testing import OCLTestCase

N = 100


def simple_smem(ary):
    sm = ocl.shared.array(N, int32)
    i = ocl.get_global_id(0)
    if i == 0:
        for j in range(N):
            sm[j] = j
    ocl.barrier()
    ary[i] = sm[i]


S0 = 10
S1 = 20


def coop_smem2d(ary):
    i, j = ocl.grid(2)
    sm = ocl.shared.array((S0, S1), float32)
    sm[i, j] = (i + 1) / (j + 1)
    ocl.barrier()
    ary[i, j] = sm[i, j]


class TestOclTestGlobal(OCLTestCase):
    def test_global_int_const(self):
        """Test simple_smem
        """
        compiled = ocl.jit("void(int32[:])")(simple_smem)

        nelem = 100
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))

    @unittest.SkipTest
    def test_global_tuple_const(self):
        """Test coop_smem2d
        """
        compiled = ocl.jit("void(float32[:,:])")(coop_smem2d)

        shape = 10, 20
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape](ary)

        exp = np.empty_like(ary)
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = float(i + 1) / (j + 1)
        self.assertTrue(np.allclose(ary, exp))


if __name__ == '__main__':
    unittest.main()
