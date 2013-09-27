import numpy as np
from .support import testcase, main, assertTrue
from numbapro import cuda
from numbapro import cudapy
from numbapro.npm.types import float32, arraytype

def add2f(a, b):
    return a + b

def indirect(a, b):
    return cadd2f(a, b)

def use_add2f(ary):
    i = cuda.grid(1)
    ary[i] = cadd2f(ary[i], ary[i])

def indirect_add2f(ary):
    i = cuda.grid(1)
    ary[i] = cindirect(ary[i], ary[i])

#------------------------------------------------------------------------------
# test_use_add2f

@testcase
def test_use_add2f():
    cadd2f = cudapy.compile_device(add2f, float32, [float32, float32], inline=True)
    globals()['cadd2f'] = cadd2f

    compiled = cudapy.compile_kernel(use_add2f, [arraytype(float32, 1, 'C')])
    compiled.bind()

    nelem = 10
    ary = np.arange(nelem, dtype=np.float32)
    exp = ary + ary
    compiled[1, nelem](ary)

    assertTrue(np.all(ary == exp), (ary, exp))

#------------------------------------------------------------------------------
# test_indirect_add2f

@testcase
def test_indirect_add2f():
    cadd2f = cudapy.compile_device(add2f, float32, [float32, float32],
                                   inline=True)
    globals()['cadd2f'] = cadd2f

    cindirect = cudapy.compile_device(indirect, float32, [float32, float32],
                                   inline=True)
    globals()['cindirect'] = cindirect


    compiled = cudapy.compile_kernel(indirect_add2f, [arraytype(float32, 1, 'C')])
    compiled.bind()

    nelem = 10
    ary = np.arange(nelem, dtype=np.float32)
    exp = ary + ary
    compiled[1, nelem](ary)

    assertTrue(np.all(ary == exp), (ary, exp))


if __name__ == '__main__':
    main()
