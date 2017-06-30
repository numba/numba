from __future__ import print_function, absolute_import
import numpy as np
from numba import ocl, int32, float32
from numba.ocl.testing import unittest
from numba.ocl.testing import OCLTestCase

def useless_sync(ary):
    i = ocl.get_global_id(0)
    ocl.barrier()
    ary[i] = i


def simple_smem(ary):
    N = 100
    sm = ocl.shared.array(N, int32)
    i = ocl.get_global_id(0)
    if i == 0:
        for j in range(N):
            sm[j] = j
    ocl.barrier()
    ary[i] = sm[i]


def coop_smem2d(ary):
    i = ocl.get_global_id(0)
    j = ocl.get_global_id(1)
    sm = ocl.shared.array((10, 20), float32)
    sm[i, j] = (i + 1) / (j + 1)
    ocl.barrier()
    ary[i, j] = sm[i, j]


def dyn_shared_memory(ary):
    i = ocl.get_global_id(0)
    sm = ocl.shared.array(0, float32)
    sm[i] = i * 2
    ocl.barrier()
    ary[i] = sm[i]


def use_threadfence(ary):
    ary[0] += 123
    ocl.threadfence()
    ary[0] += 321

@unittest.skip
def use_threadfence_block(ary):
    ary[0] += 123
    ocl.threadfence_block()
    ary[0] += 321

@unittest.skip
def use_threadfence_system(ary):
    ary[0] += 123
    ocl.threadfence_system()
    ary[0] += 321


class TestOclSync(OCLTestCase):
    def test_useless_sync(self):
        compiled = ocl.jit("void(int32[::1])")(useless_sync)
        nelem = 10
        ary = np.empty(nelem, dtype=np.int32)
        exp = np.arange(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == exp))

    def test_simple_smem(self):
        compiled = ocl.jit("void(int32[::1])")(simple_smem)
        nelem = 100
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))

    def test_coop_smem2d(self):
        compiled = ocl.jit("void(float32[:,::1])")(coop_smem2d)
        shape = 10, 20
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape](ary)
        exp = np.empty_like(ary)
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = (i + 1) / (j + 1)
        self.assertTrue(np.allclose(ary, exp))

    @unittest.skip("dynamic shared mem in OpenCL?")
    def test_dyn_shared_memory(self):
        compiled = ocl.jit("void(float32[::1])")(dyn_shared_memory)
        shape = 50
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape, 0, ary.size * 4](ary)
        self.assertTrue(np.all(ary == 2 * np.arange(ary.size, dtype=np.int32)))

    @unittest.skip("no threadfence in OpenCL?")
    def test_threadfence_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = ocl.jit("void(int32[:])")(use_threadfence)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        self.assertIn("membar.gl;", compiled.ptx)

    @unittest.skip("no threadfence-block in OpenCL?")
    def test_threadfence_block_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = ocl.jit("void(int32[:])")(use_threadfence_block)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        self.assertIn("membar.cta;", compiled.ptx)

    @unittest.skip("no threadfence-system in OpenCL?")
    def test_threadfence_system_codegen(self):
        # Does not test runtime behavior, just the code generation.
        compiled = ocl.jit("void(int32[:])")(use_threadfence_system)
        ary = np.zeros(10, dtype=np.int32)
        compiled[1, 1](ary)
        self.assertEqual(123 + 321, ary[0])
        self.assertIn("membar.sys;", compiled.ptx)


if __name__ == '__main__':
    unittest.main()
