from __future__ import print_function

import numpy as np

from numba import cuda, config
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim


def this_grid(A):
    A[0] = cuda.this_grid()


def sync_group(A):
    g = cuda.this_grid()
    cuda.sync_group(g)
    A[0] = g


if not config.ENABLE_CUDASIM:
    from numba.cuda.cudadrv.libs import get_cudalib

    LINK = get_cudalib('cudadevrt', platform='linux-static')
    assert LINK


@skip_on_cudasim("can't link on sim")
class TestCudaCooperativeGroups(SerialMixin, unittest.TestCase):
    def test_this_grid(self):
        tg = cuda.jit(
            'void(float64[:])',
            link=LINK,
        )(this_grid)
        A = np.full(1, fill_value=np.nan)
        tg.configure(1, 1, cooperative=True)(A)
        self.assertFalse(
            np.isnan(A[0]),
            'set it to something!')

    def test_sync_group(self):
        tg = cuda.jit(
            'void(float64[:])',
            link=LINK,
        )(sync_group)
        A = np.full(1, fill_value=np.nan)
        tg.configure(1, 1, cooperative=True)(A)
        self.assertFalse(
            np.isnan(A[0]),
            'set it to something!')


if __name__ == '__main__':
    unittest.main()
