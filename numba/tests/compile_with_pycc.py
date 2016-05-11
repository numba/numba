import cmath

import numpy as np

from numba import float32
from numba.pycc import CC, exportmany, export
from numba.tests.matmul_usecase import has_blas


#
# New API
#

cc = CC('pycc_test_simple')
cc.use_nrt = False

@cc.export('multf', (float32, float32))
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b

# Test imported C globals such as Py_None, PyExc_ZeroDivisionError
@cc.export('get_none', 'none()')
def get_none():
    return None

@cc.export('div', 'f8(f8, f8)')
def div(x, y):
    return x / y

_two = 2

# This one can't be compiled by the legacy API as it doesn't execute
# the script in a proper module.
@cc.export('square', 'i8(i8)')
def square(u):
    return u ** _two

# These ones need helperlib
cc_helperlib = CC('pycc_test_helperlib')
cc_helperlib.use_nrt = False

@cc_helperlib.export('power', 'i8(i8, i8)')
def power(u, v):
    return u ** v

@cc_helperlib.export('sqrt', 'c16(c16)')
def sqrt(u):
    return cmath.sqrt(u)

@cc_helperlib.export('size', 'i8(f8[:])')
def size(arr):
    return arr.size


# This one clashes with libc random() unless pycc is careful with naming.
@cc_helperlib.export('random', 'f8(i4)')
def random_impl(seed):
    if seed != -1:
        np.random.seed(seed)
    return np.random.random()

# These ones need NRT
cc_nrt = CC('pycc_test_nrt')

@cc_nrt.export('zero_scalar', 'f8(i4)')
def zero_scalar(n):
    arr = np.zeros(n)
    return arr[-1]

if has_blas:
    # This one also needs BLAS
    @cc_nrt.export('vector_dot', 'f8(i4)')
    def vector_dot(n):
        a = np.linspace(1, n, n)
        return np.dot(a, a)

# This one needs an environment
@cc_nrt.export('zeros', 'f8[:](i4)')
def zeros(n):
    return np.zeros(n)


#
# Legacy API
#

exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
# Needs to link to helperlib to due with complex arguments
# export('multc c16(c16,c16)')(mult)
export('mult f8(f8, f8)')(mult)
