from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import roc
from numba.errors import TypingError
import operator as oper

_WAVESIZE = roc.get_context().agent.wavefront_size

@roc.jit(device=True)
def shuffle_up(val, width):
    tid = roc.get_local_id(0)
    roc.wavebarrier()
    idx = (tid + width) % _WAVESIZE
    res = roc.ds_permute(idx, val)
    return res

@roc.jit(device=True)
def shuffle_down(val, width):
    tid = roc.get_local_id(0)
    roc.wavebarrier()
    idx = (tid - width) % _WAVESIZE
    res = roc.ds_permute(idx, val)
    return res

@roc.jit(device=True)
def broadcast(val, from_lane):
    tid = roc.get_local_id(0)
    roc.wavebarrier()
    res = roc.ds_bpermute(from_lane, val)
    return res

def gen_kernel(shuffunc):
    @roc.jit
    def kernel(inp, outp, amount):
        tid = roc.get_local_id(0)
        val = inp[tid]
        outp[tid] = shuffunc(val, amount)
    return kernel


class TestDsPermute(unittest.TestCase):

    def test_ds_permute(self):

        inp = np.arange(_WAVESIZE).astype(np.int32)
        outp = np.zeros_like(inp)

        for shuffler, op in [(shuffle_down, oper.neg), (shuffle_up, oper.pos)]:
            kernel = gen_kernel(shuffler)
            for shuf in range(-_WAVESIZE, _WAVESIZE):
                kernel[1, _WAVESIZE](inp, outp, shuf)
                np.testing.assert_allclose(outp, np.roll(inp, op(shuf)))

    def test_ds_permute_random_floats(self):

        inp = np.linspace(0, 1, _WAVESIZE).astype(np.float32)
        outp = np.zeros_like(inp)

        for shuffler, op in [(shuffle_down, oper.neg), (shuffle_up, oper.pos)]:
            kernel = gen_kernel(shuffler)
            for shuf in range(-_WAVESIZE, _WAVESIZE):
                kernel[1, _WAVESIZE](inp, outp, shuf)
                np.testing.assert_allclose(outp, np.roll(inp, op(shuf)))

    def test_ds_permute_type_safety(self):
        """ Checks that float64's are not being downcast to float32"""
        kernel = gen_kernel(shuffle_down)
        inp = np.linspace(0, 1, _WAVESIZE).astype(np.float64)
        outp = np.zeros_like(inp)
        with self.assertRaises(TypingError) as e:
            kernel[1, _WAVESIZE](inp, outp, 1)
        errmsg = e.exception.msg
        self.assertIn('Invalid use of Function', errmsg)
        self.assertIn('with argument(s) of type(s): (float64, int64)', errmsg)

    def test_ds_bpermute(self):

        @roc.jit
        def kernel(inp, outp, lane):
            tid = roc.get_local_id(0)
            val = inp[tid]
            outp[tid] = broadcast(val, lane)

        inp = np.arange(_WAVESIZE).astype(np.int32)
        outp = np.zeros_like(inp)
        for lane in range(0, _WAVESIZE):
            kernel[1, _WAVESIZE](inp, outp, lane)
            np.testing.assert_allclose(outp, lane)

    def test_ds_bpermute_random_floats(self):

        @roc.jit
        def kernel(inp, outp, lane):
            tid = roc.get_local_id(0)
            val = inp[tid]
            outp[tid] = broadcast(val, lane)

        inp = np.linspace(0, 1, _WAVESIZE).astype(np.float32)
        outp = np.zeros_like(inp)

        for lane in range(0, _WAVESIZE):
            kernel[1, _WAVESIZE](inp, outp, lane)
            np.testing.assert_allclose(outp, inp[lane])


if __name__ == '__main__':
    unittest.main()
