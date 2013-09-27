import numpy as np
from ..compiler import compile
from ..types import int32
from .support import testcase, main, assertTrue

def whileloop1(a, b):
    i = 0
    c = 1
    while i < a:
        i += 1
        c = b + i
    return i * c

def whileloop2(a, b):
    i = 0
    if i < b:
        while i < a:
            i += 1
    return i


def whileloop3(a, b):
    i = 0
    if i < b:
        while i < a:
            if i < b:
                i += 1
    return i

def whileloop4(a, b):
    i = 0
    if i < b:
        while i < a:
            if i < b:
                i += 1
                continue
            else:
                break
    return i


def template(func, compiled, args, allclose=False):
    got = compiled(*args)
    exp = func(*args)

    msg = '%s%s got = %s expect=%s' % (func, args, got, exp)

    if allclose:
        assertTrue(np.allclose(got, exp), msg)
    else:
        assertTrue(got == exp, msg)

#------------------------------------------------------------------------------
# whileloop1

@testcase
def test_whileloop1_integer():
    func = whileloop1
    cfunc = compile(func, int32, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

    args = 12, 50
    template(func, cfunc, args)

    args = -100, 1000
    template(func, cfunc, args)


#------------------------------------------------------------------------------
# whileloop2

@testcase
def test_whileloop2_integer():
    func = whileloop2
    cfunc = compile(func, int32, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

    args = 12, 50
    template(func, cfunc, args)

    args = -100, 1000
    template(func, cfunc, args)

#------------------------------------------------------------------------------
# whileloop3

@testcase
def test_whileloop3_integer():
    func = whileloop3
    cfunc = compile(func, int32, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

    args = 12, 50
    template(func, cfunc, args)

    args = -100, 1000
    template(func, cfunc, args)

#------------------------------------------------------------------------------
# whileloop4

@testcase
def test_whileloop4_integer():
    func = whileloop4
    cfunc = compile(func, int32, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

    args = 12, 50
    template(func, cfunc, args)

    args = -100, 1000
    template(func, cfunc, args)




if __name__ == '__main__':
    main()

