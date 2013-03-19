#! /usr/bin/env python
# ______________________________________________________________________

from numba import uint16
from numba.decorators import jit

import unittest
import __builtin__

# ______________________________________________________________________

def rshift (a, b):
    return a >> b

# ______________________________________________________________________

class TestRshift (unittest.TestCase):
    #Test for issue #152
    def test_rshift_uint16 (self):
        compiled_rshift = jit(argtypes = (uint16, uint16), 
            restype = uint16)(rshift)
        self.assertEqual(rshift(65535, 2),
                compiled_rshift(65535, 2))
                        

# ______________________________________________________________________

if __name__ == '__main__':
    unittest.main()

# ______________________________________________________________________
# End of test_rshift.py
