import numpy
from .support import testcase, main, assertTrue
from numbapro import cuda


@cuda.jit('void(double[:])')
def kernel(x):
    i = cuda.grid(1)
    if i < x.shape[0]:
        x[i] = i

@cuda.jit('void(double[:], double[:])')
def copykernel(x, y):
    i = cuda.grid(1)
    if i < x.shape[0]:
        x[i] = i
        y[i] = i

@testcase
def test_gpu_array_strided():
    x = numpy.arange(10, dtype=numpy.double)
    y = numpy.ndarray(shape=10*8, buffer=x, dtype=numpy.byte)
    z = numpy.ndarray(9, buffer=y[4:-4], dtype=numpy.double)
    kernel[10,10](z)
    assertTrue(numpy.allclose(z, list(range(9))))

@testcase
def test_gpu_array_interleaved():
    x = numpy.arange(10, dtype=numpy.double)
    y = x[:-1:2]
    z = x[1::2]
    n = y.size
    try:
        cuda.devicearray.auto_device(y)
    except ValueError:
        pass
    else:
        raise AssertionError("Should raise exception complaining the "
                             "contiguous-ness of the array.")
    # Should we handle this use case?
    # assert z.size == y.size
    # copykernel[1, n](y, x)
    # print(y, z)
    # assert numpy.all(y == z)
    # assert numpy.all(y == list(range(n)))



if __name__ == '__main__':
    main()
