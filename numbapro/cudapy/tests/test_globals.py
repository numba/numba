from __future__ import division
import numpy as np

from .support import testcase, main
from numbapro import cuda
from numbapro import cudapy
from numbapro.npm.types import int32, arraytype, float32
import numbapro

N = 100
def simple_smem(ary):
    sm = cuda.shared.array(N, numbapro.int32)
    i = cuda.grid(1)
    if i == 0:
        for j in range(N):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]

S0 = 10
S1 = 20
def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared.array((S0, S1), numbapro.float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]

#------------------------------------------------------------------------------
# simple_smem

@testcase
def test_global_int_const():
    compiled = cudapy.compile_kernel(simple_smem, [arraytype(int32, 1, 'C')])
    compiled.bind()

    nelem = 100
    ary = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary)

    assert np.all(ary == np.arange(nelem, dtype=np.int32))

#------------------------------------------------------------------------------
# coop_smem2d

@testcase
def test_global_tuple_const():
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

if __name__ == '__main__':
    main()
