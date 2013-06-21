import numpy as np

from .support import testcase, main
from numbapro import cuda
from numbapro import cudapy
from numbapro.npm.types import *

def simple_threadidx(ary):
    i = cuda.threadIdx.x
    ary[0] = i

def fill_threadidx(ary):
    i = cuda.threadIdx.x
    ary[i] = i

def fill3d_threadidx(ary):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z

    ary[i, j, k] = (i + 1) * (j + 1) * (k + 1)

def simple_grid1d(ary):
    i = cuda.grid(1)
    ary[i] = i

def simple_grid2d(ary):
    i, j = cuda.grid(2)
    ary[i, j] = i + j

#------------------------------------------------------------------------------
# simple_threadidx

@testcase
def test_simple_threadidx():
    compiled = cudapy.compile_kernel(simple_threadidx,
                                     [arraytype(int32, 1, 'C')])
    compiled.bind()

    ary = np.ones(1, dtype=np.int32)
    compiled(ary)
    assert ary[0] == 0


#------------------------------------------------------------------------------
# fill_threadidx

@testcase
def test_fill_threadidx():
    compiled = cudapy.compile_kernel(fill_threadidx,
                                     [arraytype(int32, 1, 'C')])
    compiled.bind()

    N = 10
    ary = np.ones(N, dtype=np.int32)
    exp = np.arange(N, dtype=np.int32)
    compiled[1, N](ary)
    assert np.all(ary == exp)

#------------------------------------------------------------------------------
# fill3d_threadidx

@testcase
def test_fill3d_threadidx():
    X, Y, Z = 4, 5, 6
    def c_contigous():
        compiled = cudapy.compile_kernel(fill3d_threadidx,
                                         [arraytype(int32, 3, 'C')])
        compiled.bind()

        ary = np.zeros((X, Y, Z), dtype=np.int32)
        compiled[1, (X, Y, Z)](ary)
        return ary

    def f_contigous():
        compiled = cudapy.compile_kernel(fill3d_threadidx,
                                         [arraytype(int32, 3, 'F')])
        compiled.bind()

        ary = np.asfortranarray(np.zeros((X, Y, Z), dtype=np.int32))
        compiled[1, (X, Y, Z)](ary)
        return ary

    c_res = c_contigous()
    f_res = f_contigous()
    assert np.all(c_res == f_res)

#------------------------------------------------------------------------------
# simple_grid1d

@testcase
def test_simple_grid1d():
    compiled = cudapy.compile_kernel(simple_grid1d,
                                     [arraytype(int32, 1, 'C')])
    compiled.bind()
    ntid, nctaid = 3, 7
    nelem = ntid * nctaid
    ary = np.empty(nelem, dtype=np.int32)
    compiled[nctaid, ntid](ary)
    assert np.all(ary == np.arange(nelem))

#------------------------------------------------------------------------------
# simple_grid2d

@testcase
def test_simple_grid2d():
    compiled = cudapy.compile_kernel(simple_grid2d,
                                     [arraytype(int32, 2, 'C')])
    compiled.bind()
    ntid = (4, 3)
    nctaid = (5, 6)
    shape = (ntid[0] * nctaid[0], ntid[1] * nctaid[1])
    ary = np.empty(shape, dtype=np.int32)
    exp = ary.copy()
    compiled[nctaid, ntid](ary)

    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            exp[i, j] = i + j

    assert np.all(ary == exp)

if __name__ == '__main__':
    main()
