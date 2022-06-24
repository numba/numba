import unittest

from numba import njit
from numba.tests.support import TestCase
from numba.core.opt_info import RawOptimizationRemarks


class TestOptimizationInfo(TestCase):

    def test_usage(self):
        @njit(opt_info=[RawOptimizationRemarks()])
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        foo(3)
        self.assertTrue(all('raw' in metadata['opt_info'] for metadata in
                            foo.get_metadata().values()))
        # Generate a new signature and make sure that everything still has info
        foo(3.0)
        self.assertTrue(all('raw' in metadata['opt_info'] for metadata in
                            foo.get_metadata().values()))


if __name__ == "__main__":
    unittest.main()
