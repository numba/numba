from __future__ import print_function
from numbapro import cuda
import numpy
from .support import testcase, main, assertTrue


@cuda.autojit
def foo(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


@testcase
def test_forall():
    arr = numpy.arange(11)
    orig = arr.copy()
    foo.forall(arr.size)(arr)
    assertTrue(numpy.all(arr == orig + 1))


if __name__ == '__main__':
    main()

