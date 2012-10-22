#! /usr/bin/env python
# ______________________________________________________________________
'''test_if

Test phi node (or similar) generation for CFG joins beyond
if-then-else statements.
'''
# ______________________________________________________________________

from numba import *

import unittest

# ______________________________________________________________________

def if_fn_1(arg):
    if arg > 0.:
        result = 22.
    else:
        result = 42.
    return result


def if_fn_2(i, j):
    n = 5
    m = 5
    if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
        return i + j
    return 0


# ______________________________________________________________________

class TestIf(unittest.TestCase):
    def test_if_fn_1(self):
        if_fn_1c = jit(restype=f4, argtypes=[f4], backend='ast')(if_fn_1)
        oracle = if_fn_1
        self.assertEqual(if_fn_1c(-1.), if_fn_1(-1.))
        self.assertEqual(if_fn_1c(1.),  if_fn_1(1.))

    def test_if_fn_2(self):
        if_fn_2c = jit(restype=void, argtypes=[i4, i4], backend='ast')(if_fn_2)

        oracle = if_fn_2
        for i in range(6):
            for j in range(6):
                self.assertEqual(if_fn_2c(i, j), oracle(i, j))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_if.py

