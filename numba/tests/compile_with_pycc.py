from numba import exportmany, export
from numba.pycc import CC


# New API

cc = CC('pycc_test_output')

@cc.export('multf', 'f4(f4, f4)')
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b


# Legacy API

exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
# Needs to link to helperlib to due with complex arguments
# export('multc c16(c16,c16)')(mult)
export('mult f8(f8, f8)')(mult)
