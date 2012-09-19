from numbapro.decorators import export
from numba import *

@export(arg_types=[d, d], ret_type=d)
def mult(a, b):
    return a * b

