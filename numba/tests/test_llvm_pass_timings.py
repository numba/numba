import unittest

from numba import njit
from numba.tests.support import TestCase
from numba.misc import llvm_pass_timings as lpt


class TestLLVMPassTimings(TestCase):
    def test_usage(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        foo(10)

        md = foo.get_metadata(foo.signatures[0])
        timings = md['llvm_pass_timings']
        # Check: timing is of correct type
        self.assertIsInstance(timings, lpt.PassTimingCollection)
        # Check: basic for __str__
        text = str(timings)
        self.assertIn("== Module passes (full)", text)
        # Check: there must be more than one record
        self.assertGreater(len(timings), 0)
        # Check: __getitem__
        last = timings[-1]
        self.assertIsInstance(last, lpt._NamedTimings)
        # Check: _NamedTimings
        self.assertIsInstance(last.name, str)
        self.assertIsInstance(last.timings, lpt._ProcessedPassTimings)


if __name__ == "__main__":
    unittest.main()
