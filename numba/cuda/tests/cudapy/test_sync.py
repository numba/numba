from __future__ import print_function, absolute_import
import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import unittest, SerialMixin
from numba.config import ENABLE_CUDASIM


def useless_sync(ary):
    i = cuda.grid(1)
    cuda.syncthreads()
    ary[i] = i


def simple_smem(ary):
    N = 100
    sm = cuda.shared.array(N, int32)
    i = cuda.grid(1)
    if i == 0:
        for j in range(N):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]


def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared.array((10, 20), float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]


def dyn_shared_memory(ary):
    i = cuda.grid(1)
    sm = cuda.shared.array(0, float32)
    sm[i] = i * 2
    cuda.syncthreads()
    ary[i] = sm[i]


def use_threadfence(ary):
    ary[0] += 123
    cuda.threadfence()
    ary[0] += 321


def use_threadfence_block(ary):
    ary[0] += 123
    cuda.threadfence_block()
    ary[0] += 321


def use_threadfence_system(ary):
    ary[0] += 123
    cuda.threadfence_system()
    ary[0] += 321


def use_syncthreads_count(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_count(ary_in[i])


def use_syncthreads_and(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_and(ary_in[i])


def use_syncthreads_or(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_or(ary_in[i])



class TestCudaSync(SerialMixin, unittest.TestCase):
    def test_useless_sync(self):
        compiled = cuda.jit("void(int32[::1])")(useless_sync)
        nelem = 10
        ary = np.empty(nelem, dtype=np.int32)
        exp = np.arange(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == exp))

    def test_simple_smem(self):
        compiled = cuda.jit("void(int32[::1])")(simple_smem)
        nelem = 100
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))

    def test_coop_smem2d(self):
        compiled = cuda.jit("void(float32[:,::1])")(coop_smem2d)
        shape = 10, 20
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape](ary)
        exp = np.empty_like(ary)
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = (i + 1) / (j + 1)
        self.assertTrue(np.allclose(ary, exp))

    def test_dyn_shared_memory(self):
        compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory)
        shape = 50
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape, 0, ary.size * 4](ary)
        self.assertTrue(np.all(ary == 2 * np.arange(ary.size, dtype=np.int32)))

    def test_threadfence_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = cuda.jit("void(int32[:])")(use_threadfence)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        if not ENABLE_CUDASIM:
            self.assertIn("membar.gl;", compiled.ptx)

    def test_threadfence_block_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = cuda.jit("void(int32[:])")(use_threadfence_block)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        if not ENABLE_CUDASIM:
            self.assertIn("membar.cta;", compiled.ptx)

    def test_threadfence_system_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = cuda.jit("void(int32[:])")(use_threadfence_system)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        if not ENABLE_CUDASIM:
            self.assertIn("membar.sys;", compiled.ptx)

    def test_syncthreads_count(self):
        compiled = cuda.jit("void(int32[:], int32[:])")(use_syncthreads_count)
        ary_in = np.ones(72, dtype=np.int32)
        ary_out = np.zeros(72, dtype=np.int32)
        ary_in[31] = 0
        ary_in[42] = 0
        compiled[1, 72](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 70))

    def test_syncthreads_and(self):
        compiled = cuda.jit("void(int32[:], int32[:])")(use_syncthreads_and)
        nelem = 100
        ary_in = np.ones(nelem, dtype=np.int32)
        ary_out = np.zeros(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))
        ary_in[31] = 0
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))

    def test_syncthreads_or(self):
        compiled = cuda.jit("void(int32[:], int32[:])")(use_syncthreads_or)
        nelem = 100
        ary_in = np.zeros(nelem, dtype=np.int32)
        ary_out = np.zeros(nelem, dtype=np.int32)
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 0))
        ary_in[31] = 1
        compiled[1, nelem](ary_in, ary_out)
        self.assertTrue(np.all(ary_out == 1))


if __name__ == '__main__':
    unittest.main()
