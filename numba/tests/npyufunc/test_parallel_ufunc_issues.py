from __future__ import print_function, absolute_import, division

import time
import ctypes

import numpy as np

from numba import unittest_support as unittest
from numba.tests.support import captured_output
from numba import vectorize


class TestParUfuncIssues(unittest.TestCase):
    def test_thread_response(self):
        """
        Related to #89.
        This does not test #89 but tests the fix for it.
        We want to make sure the worker threads can be used multiple times
        and with different time gap between each execution.
        """

        @vectorize('float64(float64, float64)', target='parallel')
        def fnv(a, b):
            return a + b

        sleep_time = 1   # 1 second
        while sleep_time > 0.00001:    # 10us
            time.sleep(sleep_time)
            a = b = np.arange(10**5)
            np.testing.assert_equal(a + b, fnv(a, b))
            # Reduce sleep time
            sleep_time /= 2

    def test_gil_reacquire_deadlock(self):
        """
        Testing issue #1998 due to GIL reacquiring
        """
        # make a ctypes callback that requires the GIL
        proto = ctypes.CFUNCTYPE(None, ctypes.c_int32)

        def bar(x):
            print('c', x)

        cbar = proto(bar)

        # our unit under test
        @vectorize(['int32(int32)'], target='parallel', nopython=True)
        def foo(x):
            print('p', x)  # this reacquires the GIL
            cbar(x)        # this reacquires the GIL
            return x

        # Numpy ufunc has a heuristic to determine whether to release the GIL
        # during execution.  Small input size (10) seems to not release the GIL.
        # Large input size (1000) seems to release the GIL.
        for nelem in [1, 10, 100, 1000]:
            # inputs
            a = np.arange(nelem, dtype=np.int32)
            acopy = a.copy()
            # run and capture stdout
            with captured_output('stdout') as buf:
                got = foo(a)
            stdout = buf.getvalue()
            buf.close()
            got_output = sorted(stdout.splitlines())
            # build expected output
            expected_output = list(map('c {0}'.format, range(nelem)))
            expected_output += list(map('p {0}'.format, range(nelem)))
            expected_output = sorted(expected_output)
            # verify
            self.assertEqual(got_output, expected_output)
            np.testing.assert_equal(got, acopy)


if __name__ == '__main__':
    unittest.main()
