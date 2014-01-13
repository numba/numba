#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

import unittest
from numba import autojit
import numpy as np

# ______________________________________________________________________

def _for_loop_fn_0 (iterable):
    acc = 0.
    for value in iterable:
        acc += value
    return acc

# ______________________________________________________________________

def _for_loop_fn_1 (start, stop, inc):
    acc = 0
    for value in range(start, stop, inc):
        acc += value
    return acc

@autojit
def for_loop_fn_1a (start, stop):
    acc = 0
    for value in range(start, stop):
        acc += value
    return acc

@autojit
def for_loop_fn_1b (stop):
    acc = 0
    for value in range(stop):
        acc += value
    return acc

# ______________________________________________________________________

def _for_loop_fn_2 (stop):
    acc = 0
    for value_0 in range(stop):
        for value_1 in range(stop):
            acc += value_0 * value_1
    return acc

# ______________________________________________________________________

def _for_loop_fn_3 (stop):
    acc = 0
    for i in range(stop):
        for j in range(stop):
            for k in range(stop):
                for l in range(stop):
                    acc += 1
    return acc

for_loop_fn_0 = autojit(backend='ast')(_for_loop_fn_0)
for_loop_fn_1 = autojit(backend='ast')(_for_loop_fn_1)
for_loop_fn_2 = autojit(backend='ast')(_for_loop_fn_2)
for_loop_fn_3 = autojit(backend='ast')(_for_loop_fn_3)

# ______________________________________________________________________

class TestForLoop(unittest.TestCase):
#    @unittest.skipUnless(__debug__, "Requires implementation of iteration "
#                         "over arrays.")
    def test_compiled_for_loop_fn_0(self):
        for dtype in (np.float32, np.float64, np.int32, np.int64):
            test_data = np.arange(10, dtype=dtype)
            result = for_loop_fn_0(test_data)
            self.assertEqual(result, 45)
            self.assertEqual(result, _for_loop_fn_0(test_data))

    def test_compiled_for_loop_fn_1(self):
        result = for_loop_fn_1(1, 4, 1)
        self.assertEqual(result, 6)
        self.assertEqual(result, _for_loop_fn_1(1, 4, 1))

    def test_compiled_for_loop_fn_2(self):
        result = for_loop_fn_2(4)
        self.assertEqual(result, 36)
        self.assertEqual(result, _for_loop_fn_2(4))

    def test_compiled_for_loop_fn_3(self):
        result = for_loop_fn_3(3)
        self.assertEqual(result, _for_loop_fn_3(3))
        self.assertEqual(result, 81)

    def test_compiled_for_loop_fn_many(self):
        for lo in xrange( -10, 11 ):
            for hi in xrange( -10, 11 ):
                for step in xrange( -20, 21 ):
                    if step:
                        self.assertEqual(for_loop_fn_1(lo, hi, step),
                                         for_loop_fn_1.py_func(lo, hi, step),
                                         'failed for %d/%d/%d' % (lo, hi, step))
                        self.assertEqual(for_loop_fn_1a(lo, hi),
                                         for_loop_fn_1a.py_func(lo, hi),
                                         'failed for %d/%d' % (lo, hi))
                        self.assertEqual(for_loop_fn_1b(hi),
                                         for_loop_fn_1b.py_func(hi),
                                         'failed for %d' % hi)

# ______________________________________________________________________

if __name__ == "__main__":
    print((for_loop_fn_2(10)))
#    print for_loop_fn_2(10.0)
    unittest.main()

# ______________________________________________________________________
# End of test_forloop.py
