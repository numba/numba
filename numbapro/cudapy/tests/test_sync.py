import numpy as np

from .support import testcase, main
from numbapro import cuda
from numbapro import cudapy
from numbapro.npm.types import int32, float32, arraytype
import numbapro

def useless_sync(ary):
    i = cuda.grid(1)
    cuda.syncthreads()
    ary[i] = i

def simple_smem(ary):
    N = 100
    sm = cuda.shared.array(N, numbapro.int32)
    i = cuda.grid(1)
    if i == 0:
        for j in range(N):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]

def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared.array((10, 20), numbapro.float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]


def dyn_shared_memory(ary):
    i = cuda.grid(1)
    sm = cuda.shared.array(0, numbapro.float32)
    sm[i] = i * 2
    cuda.syncthreads()
    ary[i] = sm[i]

#------------------------------------------------------------------------------
# useless_sync

@testcase
def test_useless_sync():
    compiled = cudapy.compile_kernel(useless_sync,
                                     [arraytype(int32, 1, 'C')])
    compiled.bind()

    nelem = 10
    ary = np.empty(nelem, dtype=np.int32)
    exp = np.arange(nelem, dtype=np.int32)

    compiled[1, nelem](ary)

    assert np.all(ary == exp)

#------------------------------------------------------------------------------
# simple_smem

@testcase
def test_simple_smem():
    compiled = cudapy.compile_kernel(simple_smem, [arraytype(int32, 1, 'C')])
    compiled.bind()

    nelem = 100
    ary = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary)

    assert np.all(ary == np.arange(nelem, dtype=np.int32))

#------------------------------------------------------------------------------
# coop_smem2d

@testcase
def test_coop_smem2d():
    compiled = cudapy.compile_kernel(coop_smem2d, [arraytype(float32, 2, 'C')])
    compiled.bind()

    shape = 10, 20
    ary = np.empty(shape, dtype=np.float32)
    compiled[1, shape](ary)

    exp = np.empty_like(ary)
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            exp[i, j] = float(i + 1) / (j + 1)
    assert np.allclose(ary, exp)

#------------------------------------------------------------------------------
# test_dyn_shared_memory

@testcase
def test_dyn_shared_memory():
    compiled = cudapy.compile_kernel(dyn_shared_memory,
                                     [arraytype(float32, 1, 'C')])
    compiled.bind()

    shape = 50
    ary = np.empty(shape, dtype=np.float32)
    compiled[1, shape, 0, ary.size * 4](ary)

    assert np.all(ary == 2 * np.arange(ary.size, dtype=np.int32))

if __name__ == '__main__':
    main()
