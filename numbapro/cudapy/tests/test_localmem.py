from __future__ import print_function
import numpy
from numbapro import cuda, int32
from .support import testcase, main, assertTrue


def culocal(A, B):
    C = cuda.local.array(10, dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]

@testcase
def test_local_array():
    jculocal= cuda.jit('void(int32[:], int32[:])', debug=False)(culocal)
    assertTrue('.local' in jculocal.ptx)
    A = numpy.arange(10, dtype='int32')
    B = numpy.zeros_like(A)
    jculocal(A, B)
    assertTrue(numpy.all(A == B))

if __name__ == '__main__':
    main()
