import numpy
from numbapro import cuda
from .support import testcase, main


def cuadd(a, b):
    i = cuda.grid(1)
    a[i] += b[i]

@testcase
def test_signed_overflow():
    jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
    a = numpy.array(list(reversed(range(128))), dtype=numpy.int8)

    try:
        jitted[1, a.size](a, a)
    except RuntimeError as e:
        i = e.tid[0]
        assert int(a[i]) + int(a[i]) > 127
    else:
        raise AssertionError('expeating an exception')

@testcase
def test_signed_overflow2():
    jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
    a = numpy.array(list(range(128)), dtype=numpy.int8)

    try:
        jitted[1, a.size](a, a)
    except RuntimeError as e:
        i = e.tid[0]
        assert int(a[i]) + int(a[i]) > 127
    else:
        raise AssertionError('expeating an exception')

@testcase
def test_signed_overflow3():
    jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
    a = numpy.array(list(range(128)), dtype=numpy.int8)

    try:
        jitted[2, a.size//2](a, a)
    except RuntimeError as e:
        i = e.tid[0] + e.ctaid[0] * a.size//2
        assert int(a[i]) + int(a[i]) > 127
    else:
        raise AssertionError('expeating an exception')


if __name__ == '__main__':
    main()
