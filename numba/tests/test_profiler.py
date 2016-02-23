import numpy as np
import cProfile as profiler
import pstats
from numba import jit
from numba import unittest_support as unittest

def dot(a, b):
    sum=0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum


class TestProfiler(unittest.TestCase):

    def test_profiler(self):
        """Make sure the jit-compiled function shows up in the profile stats."""

        a = np.arange(16, dtype=np.float32)
        b = np.arange(16, dtype=np.float32)
        p = profiler.Profile()
        try:
            p.enable()
            dot(a, b)
            p.disable()
            stats = pstats.Stats(p).strip_dirs()
            self.assertIn(('test_profiler.py', 7, 'dot'), stats.stats)
        finally:
            # make sure the profiler is deactivated when this test is done so as not to
            # pollute any other tests
            p.disable()
            del p

if __name__ == '__main__':
    unittest.main()
