import numpy
from .support import testcase, main, assertTrue
from numbapro import cuda


@cuda.jit('double(double[:],int64)', device=True, inline=True)
def device_function(a,c):
  return a[c]

@cuda.jit('void(double[:],double[:])')
def kernel(x,y):
    i = cuda.grid(1)
    y[i] = device_function(x, i)

@testcase
def test_array_ary():
    x = numpy.arange(10, dtype=numpy.double)
    y = numpy.zeros_like(x)
    kernel[10,1](x, y)
    assertTrue(numpy.all(x == y))

if __name__ == '__main__':
    main()
