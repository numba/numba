from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import hsa, intp

WAVESIZE = 64
WAVESIZE_BITS = 6


@hsa.jit(device=True)
def wave_reduce(val):
    tmp = val
    tid = hsa.get_local_id(0)
    laneid = tid & (WAVESIZE - 1)

    width = WAVESIZE // 2

    while width > 0:
        hsa.wavebarrier()
        other = hsa.activelanepermute_wavewidth(tmp, laneid + width, 0, False)
        if laneid < width:
            tmp += other

        width //= 2

    # First thread has the result
    hsa.wavebarrier()
    return hsa.activelanepermute_wavewidth(tmp, 0, 0, False)


@hsa.jit
def kernel_warp_reduce(inp, out):
    idx = hsa.get_global_id(0)
    val = inp[idx]
    out[idx] = wave_reduce(val)


class TestReduction(unittest.TestCase):
    def template_wave_reduce_int(self, dtype):
        numblk = 2
        inp = np.arange(numblk * WAVESIZE, dtype=dtype)
        out = np.zeros_like(inp)
        kernel_warp_reduce[numblk, WAVESIZE](inp, out)

        np.testing.assert_equal(out[:WAVESIZE], inp[:WAVESIZE].sum())
        np.testing.assert_equal(out[WAVESIZE:], inp[WAVESIZE:].sum())

    def test_wave_reduce_intp(self):
        self.template_wave_reduce_int(np.intp)

    def test_wave_reduce_int32(self):
        self.template_wave_reduce_int(np.int32)

    def template_wave_reduce_real(self, dtype):
        numblk = 2
        inp = np.linspace(0, 1, numblk * WAVESIZE).astype(dtype)
        out = np.zeros_like(inp)
        kernel_warp_reduce[numblk, WAVESIZE](inp, out)

        np.testing.assert_allclose(out[:WAVESIZE], inp[:WAVESIZE].sum())
        np.testing.assert_allclose(out[WAVESIZE:], inp[WAVESIZE:].sum())

    def test_wave_reduce_float64(self):
        self.template_wave_reduce_real(np.float64)

    def test_wave_reduce_float32(self):
        self.template_wave_reduce_real(np.float32)


if __name__ == '__main__':
    unittest.main()
