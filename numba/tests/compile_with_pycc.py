from numba import exportmany, export
from numba.pycc import CC


# New API

cc = CC('pycc_test_output')

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

# Fails because it needs _helperlib
#@cc.export('power', 'i8(i8, i8)')
def power(u, v):
    return u ** v


# Legacy API

exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
# Needs to link to helperlib to due with complex arguments
# export('multc c16(c16,c16)')(mult)
export('mult f8(f8, f8)')(mult)
