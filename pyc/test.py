from numbapro.decorators import export
from numba import *

def mult(a, b):
    return a * b

export(argtypes=[d, d], restype=d)(mult, name='mult')
export(argtypes=[f, f], restype=f)(mult, name='multf')
export(argtypes=[int32, int32], restype=int32)(mult, name='muli')
export(argtypes=[complex128, complex128], restype=complex128)(mult, name='multc')

