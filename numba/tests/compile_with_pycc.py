from numba.decorators import export
from numba import *

def mult(a, b):
    return a * b

export(argtypes=[f8, f8], restype=f8)(mult, name='mult')
export(argtypes=[f4, f4], restype=f4)(mult, name='multf')
export(argtypes=[int32, int32], restype=int32)(mult, name='multi')
export(argtypes=[complex128, complex128], restype=complex128)(mult, name='multc')

