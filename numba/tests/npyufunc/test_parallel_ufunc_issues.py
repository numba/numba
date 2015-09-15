from __future__ import print_function, absolute_import, division

from numba import unittest_support as unittest
from numba import vectorize
import numpy as np
import time

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


if __name__ == '__main__':
    unittest.main()
