from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import roc, intp

WAVESIZE = 64

@roc.jit(device=True)
def wave_reduce(val):
    tid = roc.get_local_id(0)
    laneid = tid % WAVESIZE

    width = WAVESIZE // 2
    while width:
        if laneid < width:
            val[laneid] += val[laneid + width]
            val[laneid + width] = -1 # debug
        roc.wavebarrier()
        width = width // 2

    # First thread has the result
    roc.wavebarrier()
    return val[0]

@roc.jit
def kernel_warp_reduce(inp, out):
    idx = roc.get_group_id(0)
    val = inp[idx]
    out[idx] = wave_reduce(val)

@roc.jit
def kernel_flat_reduce(inp, out):
    out[0] = wave_reduce(inp)

class TestReduction(unittest.TestCase):

    def template_wave_reduce_int(self, dtype):
        numblk = 2
        inp = np.arange(numblk * WAVESIZE, dtype=dtype).reshape(numblk, WAVESIZE)
        inp_cpy = np.copy(inp)
        out = np.zeros((numblk,))
        kernel_warp_reduce[numblk, WAVESIZE](inp, out)

        np.testing.assert_equal(out, inp_cpy.sum(axis=1))

    def test_wave_reduce_intp(self):
        self.template_wave_reduce_int(np.intp)

    def test_wave_reduce_int32(self):
        self.template_wave_reduce_int(np.int32)

    def template_wave_reduce_real(self, dtype):
        numblk = 2
        inp = np.linspace(0, 1, numblk * WAVESIZE).astype(dtype)
        inp = inp.reshape(numblk, WAVESIZE)
        inp_cpy = np.copy(inp)
        out = np.zeros((numblk,))
        kernel_warp_reduce[numblk, WAVESIZE](inp, out)

        np.testing.assert_allclose(out, inp_cpy.sum(axis=1))

    def test_wave_reduce_float64(self):
        self.template_wave_reduce_real(np.float64)

    def test_wave_reduce_float32(self):
        self.template_wave_reduce_real(np.float32)

    def test_flat_reduce(self):
        inp = np.arange(WAVESIZE) # destroyed in kernel
        out = np.zeros((1,))
        kernel_flat_reduce[1, WAVESIZE](inp, out)
        np.testing.assert_allclose(out[0], np.arange(WAVESIZE).sum())


if __name__ == '__main__':
    unittest.main()
