#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

from numba.decorators import numba_compile

import numpy

import sys
import unittest

# ______________________________________________________________________

def for_loop_fn (iterable):
    acc = 0.
    for value in iterable:
        acc += value
    return acc

# ______________________________________________________________________

class TestForLoop(unittest.TestCase):

    def test_compiled_for_loop_fn(self):
        test_data = numpy.array([1, 2, 3], dtype = 'l')
        compiled_for_loop_fn = numba_compile(arg_types = [['l']])(for_loop_fn)
        result = compiled_for_loop_fn(test_data)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn(testdata))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_forloop.py
