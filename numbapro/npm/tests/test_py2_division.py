import sys
import numpy as np
from ..compiler import compile
from ..types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, complex64, complex128
)
from .support import testcase, main, assertTrue

def div(a, b):
    return a / b

def floordiv(a, b):
    return a // b

iset = [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
fset = [float32, float64]
cset = [complex64, complex128]

#------------------------------------------------------------------------------
# div

if sys.platform != 'win32':
    @testcase
    def test_py2_div_integer():
        def run(ty, a, b):
            cdiv = compile(div, ty, [ty, ty])
            got = cdiv(a, b)
            exp = div(a, b)
            assertTrue(got == exp,
                        msg=('div(%s, %s) got = %s expect=%s' %
                                                (a, b, got, exp)))

        tyset = set(iset)
        if tuple.__itemsize__ == 4: # 32bit
            tyset -= set([int64, uint64])
        for ty in tyset:
            run(ty, 4, 2)

@testcase
def test_py2_div_float():
    def run(ty, a, b):
        cdiv = compile(div, ty, [ty, ty])
        got = cdiv(a, b)
        exp = div(a, b)
        assertTrue(np.allclose(got, exp),
                   msg=('div(%s, %s) got = %s expect=%s' %
                                                (a, b, got, exp)))

    for ty in fset:
        run(ty, 1.234, 2.345)


#------------------------------------------------------------------------------
# floordiv

@testcase
def test_py2_floordiv_integer():
    def run(ty, a, b):
        cfloordiv = compile(floordiv, ty, [ty, ty])
        got = cfloordiv(a, b)
        exp = floordiv(a, b)
        assertTrue(got == exp, 'floordiv(%s, %s) got = %s expect=%s' % (a, b, got, exp))

    for ty in iset:
        run(ty, 4, 2)

@testcase
def test_py2_floordiv_float():
    def run(ty, a, b):
        cfloordiv = compile(floordiv, ty, [ty, ty])
        got = cfloordiv(a, b)
        exp = floordiv(a, b)
        assertTrue(got == exp, 'floordiv(%s, %s) got = %s expect=%s' % (a, b, got, exp))

    for ty in fset:
        run(ty, 1.234, 2.345)

if __name__ == '__main__':
    main()

