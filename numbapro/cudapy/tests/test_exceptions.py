import numpy
from numbapro import cuda
from .support import testcase, main, assertTrue


def cuadd(a, b):
    i = cuda.grid(1)
    a[i] += b[i]

class MyError(Exception):
    pass

def cuusererr():
    raise MyError

@testcase
def no_error():
    jitted = cuda.jit('void(int32[:], int32[:])', debug=True)(cuadd)
    a = numpy.array(list(reversed(range(128))), dtype=numpy.int32)
    jitted[1, a.size](a, a)

@testcase
def user_error():
    jitted = cuda.jit('void()', debug=True)(cuusererr)
    try:
        jitted()
    except RuntimeError as e:
        assertTrue(issubclass(e.exc, MyError))

@testcase
def test_signed_overflow():
    jitted = cuda.jit('void(int8[:], int8[:])', debug=True)(cuadd)
    a = numpy.array(list(reversed(range(128))), dtype=numpy.int8)

    try:
        jitted[1, a.size](a, a)
    except RuntimeError as e:
        i = e.tid[0]
        assertTrue(int(a[i]) + int(a[i]) > 127)
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
        assertTrue(int(a[i]) + int(a[i]) > 127)
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
        assertTrue(int(a[i]) + int(a[i]) > 127)
    else:
        raise AssertionError('expeating an exception')


if __name__ == '__main__':
    main()
