from __future__ import print_function
import numpy
from numbapro import cuda, int32, int16, float32, complex64
from .support import testcase, main, assertTrue


def cuhello():
    i = cuda.grid(1)
    print("hello", i, int32(i), i == i, i != i)
    print("yoo")
    print(int16(1024), "int16")
    print(float32(3.21), 1.23)
    print(1j, complex64(3-2j))


def cuprintary(A):
    i = cuda.grid(1)
    print("A[", i, "]", A[i])


@testcase
def test_cuhello():
    """
    Eyeballing required
    """
    jcuhello = cuda.jit('void(int32[:], int32[:])', debug=False)(cuhello)
    print(jcuhello.ptx)
    assertTrue('.const' in jcuhello.ptx)
    jcuhello[2, 3]()
    cuda.synchronize()


@testcase
def test_print_array():
    """
    Eyeballing required
    """
    jcuprintary = cuda.jit('void(float32[:])')(cuprintary)
    print(jcuprintary.ptx)
    assertTrue('.const' in jcuprintary.ptx)
    A = numpy.arange(10, dtype=numpy.float32)
    jcuprintary[2, 5](A)
    cuda.synchronize()


if __name__ == '__main__':
    main()
