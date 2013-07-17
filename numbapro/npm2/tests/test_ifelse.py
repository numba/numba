import numpy as np
from ..compiler import compile
from ..types import (
int8, int16, int32, int64, uint8, uint16, uint32, uint64,
float32, float64
)
from .support import testcase, main


iset = [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
fset = [float32, float64]

def gt(a, b):
    if a > b:
        return a
    else:
        return b

def ge(a, b):
    if a >= b:
        return a
    else:
        return b

def lt(a, b):
    if a < b:
        return a
    else:
        return b

def le(a, b):
    if a < b:
        return a
    else:
        return b

def eq(a, b):
    if a == b:
        return a
    else:
        return b

def ne(a, b):
    if a != b:
        return a
    else:
        return b

def raw(a, b):
    if a:
        return a
    else:
        return b


def template_integer(func, avalues, bvalues):
    def run(ty):
        compiled = compile(func, ty, [ty, ty])
        for a, b, in zip(avalues, bvalues):
            got = compiled(a, b)
            exp = func(a, b)
            msg = '%s(%s, %s) got = %s expect=%s' % (func, a, b, got, exp)
            assert got == exp, msg

    for ty in iset:
        run(ty)

def template_float(func, avalues, bvalues):
    def run(ty):
        compiled = compile(func, ty, [ty, ty])
        for a, b, in zip(avalues, bvalues):
            got = compiled(a, b)
            exp = func(a, b)
            msg = '%s(%s, %s) got = %s expect=%s' % (func, a, b, got, exp)
            assert np.allclose(got, exp), msg

    for ty in fset:
        run(ty)


#------------------------------------------------------------------------------
# gt

@testcase
def test_gt_integer():
    template_integer(gt, [12, 56], [34, 32])

@testcase
def test_gt_float():
    template_float(gt, [1.2, 5.6], [3.4, 3.2])


#------------------------------------------------------------------------------
# ge

@testcase
def test_ge_integer():
    template_integer(ge, [12, 56], [12, 32])

@testcase
def test_ge_float():
    template_float(ge, [1.2, 5.6], [1.2, 3.2])

#------------------------------------------------------------------------------
# lt

@testcase
def test_lt_integer():
    template_integer(lt, [12, 56], [34, 32])

@testcase
def test_lt_float():
    template_float(lt, [1.2, 5.6], [3.4, 3.2])


#------------------------------------------------------------------------------
# le

@testcase
def test_le_integer():
    template_integer(le, [12, 56], [12, 32])

@testcase
def test_le_float():
    template_float(le, [1.2, 5.6], [1.2, 3.2])


#------------------------------------------------------------------------------
# eq

@testcase
def test_eq_integer():
    template_integer(eq, [12, 56], [12, 32])

@testcase
def test_eq_float():
    template_float(eq, [1.2, 5.6], [1.2, 3.2])


#------------------------------------------------------------------------------
# ne

@testcase
def test_ne_integer():
    template_integer(ne, [12, 56], [12, 32])

@testcase
def test_ne_float():
    template_float(ne, [1.2, 5.6], [1.2, 3.2])

#------------------------------------------------------------------------------
# raw

@testcase
def test_raw_integer():
    template_integer(raw, [12, 56], [12, 32])

@testcase
def test_raw_float():
    template_float(raw, [1.2, 5.6], [1.2, 3.2])


if __name__ == '__main__':
    main()

