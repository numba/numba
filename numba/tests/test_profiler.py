import cProfile as profiler
import os
import pstats
import subprocess
import sys

import numpy as np

from numba import jit
from numba.tests.support import needs_blas
import unittest


def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum

def np_dot(a, b):
    return np.dot(a, b)


class TestProfiler(unittest.TestCase):

    def check_profiler_dot(self, pyfunc):
        """
        Make sure the jit-compiled function shows up in the profile stats
        as a regular Python function.
        """
        a = np.arange(16, dtype=np.float32)
        b = np.arange(16, dtype=np.float32)
        cfunc = jit(nopython=True)(pyfunc)
        # Warm up JIT
        cfunc(a, b)
        p = profiler.Profile()
        p.enable()
        try:
            cfunc(a, b)
        finally:
            p.disable()
        stats = pstats.Stats(p).strip_dirs()
        code = pyfunc.__code__
        expected_key = (os.path.basename(code.co_filename),
                        code.co_firstlineno,
                        code.co_name,
                        )
        self.assertIn(expected_key, stats.stats)

    def test_profiler(self):
        self.check_profiler_dot(dot)

    @needs_blas
    def test_profiler_np_dot(self):
        # Issue #1786: initializing BLAS would crash when profiling
        code = """if 1:
            import cProfile as profiler

            import numpy as np

            from numba import jit
            from numba.tests.test_profiler import np_dot

            cfunc = jit(nopython=True)(np_dot)

            a = np.arange(16, dtype=np.float32)
            b = np.arange(16, dtype=np.float32)

            p = profiler.Profile()
            p.enable()
            cfunc(a, b)
            cfunc(a, b)
            p.disable()
            """
        subprocess.check_call([sys.executable, "-c", code])

    def test_issue_3229(self):
        # Issue #3229: Seemingly random segfaults when profiling due to
        # frame injection.
        # numba.tests.npyufunc.test_dufunc.TestDUFunc.test_npm_call is the
        # first test case crashing when profiling. Fingers crossed fixing
        # this is sufficient proof for the general case.

        code = """if 1:
            import cProfile as profiler
            p = profiler.Profile()
            p.enable()

            from numba.tests.npyufunc.test_dufunc import TestDUFunc
            t = TestDUFunc('test_npm_call')
            t.test_npm_call()

            p.disable()
            """
        subprocess.check_call([sys.executable, "-c", code])

if __name__ == '__main__':
    unittest.main()
