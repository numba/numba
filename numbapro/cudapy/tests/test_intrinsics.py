import numpy as np
from .support import testcase, main, run
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
        exp = ary.copy()
        compiled[1, (X, Y, Z)](ary)
        return ary

    def f_contigous():
        compiled = cudapy.compile_kernel(fill3d_threadidx,
                                         [arraytype(int32, 3, 'F')])
        compiled.bind()

        ary = np.asfortranarray(np.zeros((X, Y, Z), dtype=np.int32))
        exp = ary.copy()
        compiled[1, (X, Y, Z)](ary)
        return ary

    c_res = c_contigous()
    f_res = f_contigous()
    assert np.all(c_res == f_res)


if __name__ == '__main__':
    main()
