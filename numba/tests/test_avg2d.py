#! /usr/bin/env python
# ______________________________________________________________________
'''Unit test for issue #30.'''
# ______________________________________________________________________
import numpy

from numba import d
from numba.decorators import jit

import unittest
import __builtin__

# ______________________________________________________________________
    
def avg2d(arr, result):
    M, N = arr.shape
    for i in range(M):
        avg = 0.
        for j in range(N):
            avg += arr[i,j]
        result[i] = avg / N

# ______________________________________________________________________

def avg2d_w_cast(arr, result):
    M, N = arr.shape
    for i in range(M):
        avg = 0.
        for j in range(N):
            avg += arr[i,j]
        result[i] = avg / float(N)

# ______________________________________________________________________

class TestAvg2D (unittest.TestCase):
    def _do_test (self, _avg2d):
        compiled_fn = jit(arg_types = [d[:,:], d[:]])(_avg2d)
        test_data = numpy.random.random((5,5))
        control_result = numpy.zeros((5,))
        test_result = control_result[:]
        _avg2d(test_data, control_result)
        compiled_fn(test_data, test_result)
        self.assertTrue((control_result == test_result).all())

    def test_avg2d (self):
        self._do_test(avg2d)

    @unittest.skipUnless(hasattr(__builtin__, '__noskip__'),
                         "Need support for float() builtin.")
    def test_avg2d_w_cast (self):
        self._do_test(avg2d_w_cast)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_avg2d.py
