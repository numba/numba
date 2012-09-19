from numbapro.decorators import export
from numba import *

def mult(a, b):
    return a * b

export(arg_types=[d, d], ret_type=d)(mult, name='mult')
export(arg_types=[f, f], ret_type=f)(mult, name='multf')

