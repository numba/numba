#!/usr/bin/env python
from numba.pycc import exportmany, export


def mult(a, b):
    return a * b

export('multi i4(i4, i4)')(mult)
exportmany(['multf f4(f4, f4)', 'mult f8(f8, f8)'])(mult)
