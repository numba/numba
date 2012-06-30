#! /usr/bin/env python
# ______________________________________________________________________
'''test_type_inference

Test type inference.
'''
# ______________________________________________________________________

from numba.decorators import function

import unittest

# ______________________________________________________________________

def _simple_func(arg):
    if arg > 0.:
        result = 22.
    else:
        result = 42.
    return result

simple_func = function(_simple_func)

def _for_loop(start, stop, inc):
    acc = 0
    for value in range(start, stop, inc):
        acc += value
    return acc

for_loop = function(_for_loop)

# ______________________________________________________________________

class TestTypeInference(unittest.TestCase):
    def test_simple_func(self):
        self.assertEqual(simple_func(-1.), 42.)
        self.assertEqual(simple_func(1.), 22.)

    def test_simple_for(self):
        self.assertEqual(for_loop(0, 10, 1), 45)

# ______________________________________________________________________

if __name__ == "__main__":
    #import dis
    # dis.dis(_simple_func)
    #dis.dis(for_loop)
    unittest.main()
