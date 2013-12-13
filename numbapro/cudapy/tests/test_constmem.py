from __future__ import print_function
import numpy
from numbapro import cuda, int32
from .support import testcase, main, assertTrue

CONST1D = numpy.arange(10, dtype=numpy.float64) / 2.
CONST2D = numpy.asfortranarray(
                numpy.arange(100, dtype=numpy.int32).reshape(10, 10))
CONST3D = ((numpy.arange(5*5*5, dtype=numpy.complex64).reshape(5, 5, 5) + 1j) /
            2j)

def cuconst(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    A[i] = C[i]


def cuconst2d(A):
    C = cuda.const.array_like(CONST2D)
    i, j = cuda.grid(2)
    A[i, j] = C[i, j]


def cuconst3d(A):
    C = cuda.const.array_like(CONST3D)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z
    A[i, j, k] = C[i, j, k]

@testcase
def test_const_array():
    jcuconst = cuda.jit('void(float64[:])')(cuconst)
    print(jcuconst.ptx)
    assertTrue('.const' in jcuconst.ptx)
    A = numpy.empty_like(CONST1D)
    jcuconst[2, 5](A)
    assertTrue(numpy.all(A == CONST1D))


@testcase
def test_const_array_2d():
    jcuconst2d = cuda.jit('void(int32[:,:])')(cuconst2d)
    print(jcuconst2d.ptx)
    assertTrue('.const' in jcuconst2d.ptx)
    A = numpy.empty_like(CONST2D, order='C')
    jcuconst2d[(2,2), (5,5)](A)
    print(CONST2D)
    print(A)
    assertTrue(numpy.all(A == CONST2D))


@testcase
def test_const_array_3d():
    jcuconst3d = cuda.jit('void(complex64[:,:,:])')(cuconst3d)
    print(jcuconst3d.ptx)
    assertTrue('.const' in jcuconst3d.ptx)
    A = numpy.empty_like(CONST3D, order='F')
    jcuconst3d[1, (5,5,5)](A)
    print(CONST3D)
    print(A)
    assertTrue(numpy.all(A == CONST3D))


if __name__ == '__main__':
    main()
