from numba import exportmany, export
from numba import decorators

def mult(a, b):
    return a * b

exportmany(['multf f4(f4,f4)', 'multi i4(i4,i4)'])(mult)
export('multc c16(c16,c16)')(mult)
export('mult f8(f8, f8)')(mult)
