from numbapro.decorators import export
from numba import *

def mult(a, b):
    return a * b

export(arg_types=[d, d], ret_type=d)(mult, name='mult')
export(arg_types=[f, f], ret_type=f)(mult, name='multf')
export(arg_types=[int32, int32], ret_type=int32)(mult, name='muli')
export(arg_types=[complex128, complex128], ret_type=complex128)(mult, name='multc')

