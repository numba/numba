#! /usr/bin/env python
# ______________________________________________________________________
'''test_if

Test phi node (or similar) generation for CFG joins beyond
if-then-else statements.
'''
# ______________________________________________________________________

from numba.decorators import jit

import unittest

# ______________________________________________________________________

def if_fn_1(arg):
    if arg > 0.:
        result = 22.
    else:
        result = 42.
    return result

# ______________________________________________________________________

class TestIf(unittest.TestCase):
    def test_if_fn_1(self):
        if_fn_1c = jit()(if_fn_1)
        self.assertEqual(if_fn_1c(-1.), 42.)
        self.assertEqual(if_fn_1c(1.), 22.)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_if.py

