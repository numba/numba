import numpy as np
from ..compiler import compile
from ..types import int32, int64, arraytype, float32, float64, void
from .support import testcase, main

def getitem(a, i):
    return a[i]

def getitem2d(a, i, j):
    return a[i, j]

def setitem(a, i, b):
    a[i] = b

def getshape(a):
    return a.shape[0]

def unpack_shape(a):
    m, n = a.shape
    return m + n * 2

def getstrides(a):
    return a.strides[0]

def getsize(a):
    return a.size

def getndim(a):
    return a.ndim

def sum1d(a):
    t = 0
    for i in range(a.size):
        t += a[i] * i
    return t

def sum2d(a):
    t = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            t += a[i, j] * (i + j)
    return t

def saxpy(a, x, y, out):
    for i in range(out.shape[0]):
        out[i] = a * x[i] + y[i]


#------------------------------------------------------------------------------
# getitem

@testcase
def test_getitem_c():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'C'), int32])
    ary = np.arange(10, dtype=np.int32)
    for i in range(ary.size):
        assert compiled(ary, i) == getitem(ary, i)

@testcase
def test_getitem_f():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'F'), int32])
    ary = np.asfortranarray(np.arange(10, dtype=np.int32))
    for i in range(ary.size):
        assert compiled(ary, i) == getitem(ary, i)

@testcase
def test_getitem_a():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'A'), int32])
    ary = np.arange(10, dtype=np.int32)
    for i in range(ary.size):
        assert compiled(ary, i) == getitem(ary, i)

#------------------------------------------------------------------------------
# getitem2d

@testcase
def test_getitem2d_c():
    compiled = compile(getitem2d, int32, [arraytype(int32, 2, 'C'), int32, int32])
    ary = np.arange(3 * 4, dtype=np.int32).reshape(3, 4)
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            got = compiled(ary, i, j)
            exp = getitem2d(ary, i, j)
            assert got == exp, (got, exp, (i, j))

@testcase
def test_getitem2d_f():
    compiled = compile(getitem2d, int32, [arraytype(int32, 2, 'F'), int32, int32])
    ary = np.asfortranarray(np.arange(3 * 4, dtype=np.int32).reshape(3, 4))
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            got = compiled(ary, i, j)
            exp = getitem2d(ary, i, j)
            print ary
            assert got == exp, (got, exp, (i, j))

@testcase
def test_getitem2d_a():
    compiled = compile(getitem2d, int32, [arraytype(int32, 2, 'A'), int32, int32])
    ary = np.asfortranarray(np.arange(3 * 4, dtype=np.int32).reshape(3, 4))
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            got = compiled(ary, i, j)
            exp = getitem2d(ary, i, j)
            assert got == exp, (got, exp, (i, j))


#------------------------------------------------------------------------------
# setitem

@testcase
def test_setitem():
    compiled = compile(setitem, void, [arraytype(int32, 1, 'C'), int32, int64])
    ary = np.arange(10, dtype=np.int32)
    orig = np.arange(10, dtype=np.int32)
    for i in range(ary.size):
        compiled(ary, i, i * 2)
    assert np.all(orig * 2 == ary)

#------------------------------------------------------------------------------
# getshape

@testcase
def test_getshape():
    compiled = compile(getshape, int32, [arraytype(int32, 1, 'C')])
    ary = np.zeros(10, dtype=np.int32)
    assert compiled(ary) == ary.shape[0]

@testcase
def test_unpack_shape():
    compiled = compile(unpack_shape, int32, [arraytype(int32, 2, 'C')])
    ary = np.empty((2, 3), dtype=np.int32)
    assert compiled(ary) == 2 + 3 * 2

#------------------------------------------------------------------------------
# getstrides

@testcase
def test_getstrides():
    compiled = compile(getstrides, int32, [arraytype(int32, 1, 'C')])
    ary = np.zeros(10, dtype=np.int32)
    assert compiled(ary) == ary.strides[0]

#------------------------------------------------------------------------------
# getsize

@testcase
def test_getsize():
    compiled = compile(getsize, int32, [arraytype(int32, 1, 'C')])
    ary = np.zeros(10, dtype=np.int32)
    assert compiled(ary) == ary.size

#------------------------------------------------------------------------------
# getndim

@testcase
def test_getndim():
    compiled = compile(getndim, int32, [arraytype(int32, 4, 'C')])
    ary = np.zeros((1, 2, 2, 4), dtype=np.int32)
    assert compiled(ary) == ary.ndim

#------------------------------------------------------------------------------
# sum1d

@testcase
def test_sum1d():
    compiled = compile(sum1d, int32, [arraytype(int32, 1, 'C')])
    ary = np.arange(10, dtype=np.int32)
    got = compiled(ary)
    exp = sum1d(ary)
    assert got == exp, (got, exp)


#------------------------------------------------------------------------------
# sum1d

@testcase
def test_sum2d_c():
    compiled = compile(sum2d, int32, [arraytype(int32, 2, 'C')])
    ary = np.arange(10, dtype=np.int32).reshape(2, 5)
    got = compiled(ary)
    exp = sum2d(ary)
    assert got == exp, (got, exp)

@testcase
def test_sum2d_f():
    compiled = compile(sum2d, int32, [arraytype(int32, 2, 'F')])
    ary = np.asfortranarray(np.arange(10, dtype=np.int32).reshape(2, 5))
    got = compiled(ary)
    exp = sum2d(ary)
    assert got == exp, (got, exp)


#------------------------------------------------------------------------------
# saxpy

@testcase
def test_saxpy():
    argtys = [int32,
              arraytype(float32, 1, 'C'),
              arraytype(float32, 1, 'C'),
              arraytype(float64, 1, 'C')]

    compiled = compile(saxpy, void, argtys)
    a = 2
    x = np.arange(10, dtype=np.float32)
    y = np.arange(10, dtype=np.float32)
    got = np.empty_like(x, dtype=np.float64)
    exp = np.empty_like(x, dtype=np.float64)

    compiled(a, x, y, got)
    saxpy(a, x, y, exp)
    assert np.allclose(got, exp), (got, exp)


#------------------------------------------------------------------------------
# getitem out-of-bound

@testcase
def test_getitem_outofbound():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'C'), int32])
    ary = np.arange(10, dtype=np.int32)
    try:
        compiled(ary, 10)
    except IndexError, e:
        print e
    else:
        raise AssertionError('expecting exception')


@testcase
def test_getitem2d_outofbound():
    compiled = compile(getitem2d, int32, [arraytype(int32, 2, 'C'), int32, int32])
    ary = np.arange(10, dtype=np.int32).reshape(2, 5)
    try:
        compiled(ary, 1, 6)
    except IndexError, e:
        print e
    else:
        raise AssertionError('expecting exception')

#------------------------------------------------------------------------------
# getitem wraparound

@testcase
def test_getitem_wraparound():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'C'), int32])
    ary = np.arange(10, dtype=np.int32)
    exp = getitem(ary, -10)
    got = compiled(ary, -10)
    assert got == exp

@testcase
def test_getitem2d_wraparound():
    compiled = compile(getitem2d, int32, [arraytype(int32, 2, 'C'), int32, int32])
    ary = np.arange(10, dtype=np.int32).reshape(2, 5)
    exp = getitem2d(ary, 1, -4)
    got = compiled(ary, 1, -4)
    assert got == exp

@testcase
def test_getitem_outofbound_overflow():
    compiled = compile(getitem, int32, [arraytype(int32, 1, 'C'), int32])
    ary = np.arange(10, dtype=np.int32)
    try:
        compiled(ary, -11)
    except IndexError, e:
        print e
    else:
        raise AssertionError('expecting exception')



if __name__ == '__main__':
    main()
