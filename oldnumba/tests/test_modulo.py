#! /usr/bin/env python
# ______________________________________________________________________

from numba import uint32, int16
from numba.decorators import jit, autojit

import unittest
import __builtin__

# ______________________________________________________________________

def modulo (a, b):
    return a % b

# ______________________________________________________________________

class TestModulo (unittest.TestCase):
    #Test for issue #143
    def test_modulo_uint32 (self):
        compiled_modulo = jit(argtypes = (uint32, uint32), 
            restype = uint32)(modulo)
        self.assertEqual(modulo(0, 0x80000000),
                compiled_modulo(0, 0x80000000))
    
    #Test for issue #151
    def test_modulo_int16 (self):
        compiled_modulo = jit(argtypes = (int16, int16), 
            restype = int16)(modulo)
        self.assertEqual(modulo(-3584, -512),
                compiled_modulo(-3584, -512))
                        

# ______________________________________________________________________

if __name__ == '__main__':
    unittest.main()

# ______________________________________________________________________
# End of test_modulo.py
