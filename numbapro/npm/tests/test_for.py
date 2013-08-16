import numpy as np
from ..compiler import compile
from ..types import int32, uint32, float32
from .support import testcase, main

def forloop1(a, b):
    j = 0
    for i in xrange(a, b):
        j += i
    return j

def forloop2(a, b):
    for i in xrange(a):
        b += i
    return b

def forloop3(a, b, c):
    i = a
    for i in xrange(a, b, c):
        pass
    return i

def forloop4(a, b, c):
    j = 0
    for i in xrange(a, b, c):
        j += i
    return j

def forloop5(a, b):
    s = 0
    for i in xrange(1, a):
        for j in range(1, b):
            s += i * j
    return s

def forloop6(a, b):
    s = 0
    for i in xrange(1, a):
        for j in range(1, b):
            s += i * j
            if s > 10:
                break
    return s

def forloop7(a, b, c):
    s = 0
    for i in xrange(a):
        for j in range(b):
            for k in range(c):
                s += i * j + k
    return s

def forloop8(a, b, c):
    s = 0
    for i in xrange(a):
        for j in range(b):
            if i + j < 10:
                for k in range(c):
                    s += i * j + k
    return s

def forloop9(a, b, c):
    j = 0
    if b > c:
        for i in xrange(a):
            j += i * b - c
    return j

def forloop10(a, b):
    s = 0
    for i in xrange(1, a):
        for j in range(1, b):
            s += i * j
            if s < 10:
                continue
            else:
                break
    return s

def template(func, compiled, args, allclose=False):
    got = compiled(*args)
    exp = func(*args)

    msg = '%s%s got = %s expect=%s' % (func, args, got, exp)

    if allclose:
        assert np.allclose(got, exp), msg
    else:
        assert got == exp, msg


#------------------------------------------------------------------------------
# forloop1

@testcase
def test_forloop1_integer():
    func = forloop1
    cfunc = compile(func, int32, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

    args = 12, 50
    template(func, cfunc, args)

    args = -100, 1000
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# forloop2

@testcase
def test_forloop2_integer():
    func = forloop2
    cfunc = compile(func, float32, [int32, float32])

    args = 0, 10.
    template(func, cfunc, args, allclose=True)

    args = 12, 50.
    template(func, cfunc, args, allclose=True)

    args = -100, 1000.
    template(func, cfunc, args, allclose=True)


#------------------------------------------------------------------------------
# forloop3

@testcase
def test_forloop3_integer():
    func = forloop3
    cfunc = compile(func, int32, [uint32, uint32, int32])

    args = 0, 10, 1
    template(func, cfunc, args)


    args = 10, 2, -1
    template(func, cfunc, args)

    args = 10, 100, 2
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# forloop4

@testcase
def test_forloop4_integer():
    func = forloop4
    cfunc = compile(func, int32, [int32, int32, int32])

    args = 0, 10, 1
    template(func, cfunc, args)


    args = 10, 2, -1
    template(func, cfunc, args)

    args = 10, 100, 2
    template(func, cfunc, args)

    args = 21, -3, -2
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# forloop5

@testcase
def test_forloop5_integer():
    func = forloop5
    cfunc = compile(func, int32, [int32, int32])

    args = 12, 34
    template(func, cfunc, args)
    
    args = 100, 100
    template(func, cfunc, args)

#------------------------------------------------------------------------------
# forloop6

@testcase
def test_forloop6_integer():
    func = forloop6
    cfunc = compile(func, int32, [int32, int32])

    args = 12, 34
    template(func, cfunc, args)
    
    args = 100, 100
    template(func, cfunc, args)

#------------------------------------------------------------------------------
# forloop7

@testcase
def test_forloop7_integer():
    func = forloop7
    cfunc = compile(func, int32, [int32, int32, int32])

    args = 1, 2, 3
    template(func, cfunc, args)
    
    args = 4, 5, 6
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# forloop8

@testcase
def test_forloop8_integer():
    func = forloop8

    cfunc = compile(func, int32, [int32, int32, int32])

    args = 1, 2, 3
    template(func, cfunc, args)
    
    args = 4, 5, 6
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# forloop9

@testcase
def test_forloop9_integer():
    func = forloop9
    cfunc = compile(func, int32, [int32, int32, int32])

    args = 1, 2, 3
    template(func, cfunc, args)
    
    args = 4, 10, 6
    template(func, cfunc, args)

#------------------------------------------------------------------------------
# forloop10

@testcase
def test_forloop10_integer():
    func = forloop10
    cfunc = compile(func, int32, [int32, int32])

    args = 1, 2
    template(func, cfunc, args)
    
    args = 4, 10
    template(func, cfunc, args)

if __name__ == '__main__':
    main()

