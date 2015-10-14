import cmath

import numpy as np

from numba import exportmany, export
from numba.pycc import CC


#
# New API
#

cc = CC('pycc_test_simple')

@cc.export('multf', 'f4(f4, f4)')
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b

_two = 2

# This one can't be compiled by the legacy API as it doesn't execute
# the script in a proper module.
@cc.export('square', 'i8(i8)')
def square(u):
    return u ** _two

# These ones need helperlib
cc_helperlib = CC('pycc_test_helperlib')

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
    np.random.seed(seed)
    return np.random.random()

# These ones need NRT
cc_nrt = CC('pycc_test_nrt')
cc_nrt.use_nrt = True

@cc_nrt.export('zero_scalar', 'f8(i4)')
def zero_scalar(n):
    arr = np.zeros(n)
    return arr[-1]

# Fails because it needs an environment
#@cc_nrt.export('zeros', 'f8[:](i4)')
#def zeros(n):
    #return np.zeros(n)


#
# Legacy API
#

exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
# Needs to link to helperlib to due with complex arguments
# export('multc c16(c16,c16)')(mult)
export('mult f8(f8, f8)')(mult)
