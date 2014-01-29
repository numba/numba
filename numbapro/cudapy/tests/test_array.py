import numpy
from .support import testcase, main, assertTrue
from numbapro import cuda


@cuda.jit('void(double[:])')
def kernel(x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        x[i] = i

@testcase
def test_gpu_array_strided():
    x = numpy.arange(10, dtype=numpy.double)
    y = numpy.ndarray(shape=10*8, buffer=x, dtype=numpy.byte)
    z = numpy.ndarray(9, buffer=y[4:-4], dtype=numpy.double)
    kernel[10,10](z)
    assertTrue(numpy.allclose(z, list(range(9))))

if __name__ == '__main__':
    main()
