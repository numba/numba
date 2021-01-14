import unittest

from numba import njit
from numba.tests.support import TestCase, override_config
from numba.misc import llvm_pass_timings as lpt


class TestLLVMPassTimings(TestCase):

    def test_usage(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        with override_config('LLVM_PASS_TIMINGS', True):
            foo(10)

        md = foo.get_metadata(foo.signatures[0])
        timings = md['llvm_pass_timings']
        # Check: timing is of correct type
        self.assertIsInstance(timings, lpt.PassTimingsCollection)
        # Check: basic for __str__
        text = str(timings)
        self.assertIn("Module passes (full optimization)", text)
        # Check: there must be more than one record
        self.assertGreater(len(timings), 0)
        # Check: __getitem__
        last = timings[-1]
        self.assertIsInstance(last, lpt.NamedTimings)
        # Check: NamedTimings
        self.assertIsInstance(last.name, str)
        self.assertIsInstance(last.timings, lpt.ProcessedPassTimings)

    def test_analyze(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                for j in range(i):
                    c += j
            return c

        with override_config('LLVM_PASS_TIMINGS', True):
            foo(10)

        md = foo.get_metadata(foo.signatures[0])
        timings_collection = md['llvm_pass_timings']
        # Check: get_total_time()
        self.assertIsInstance(timings_collection.get_total_time(), float)
        # Check: summary()
        self.assertIsInstance(timings_collection.summary(), str)
        # Check: list_longest_first() ordering
        longest_first = timings_collection.list_longest_first()
        self.assertEqual(len(longest_first), len(timings_collection))
        last = longest_first[0].timings.get_total_time()
        for rec in longest_first[1:]:
            cur = rec.timings.get_total_time()
            self.assertGreaterEqual(last, cur)
            cur = last


class TestLLVMPassTimingsDisabled(TestCase):
    def test_disabled_behavior(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        with override_config('LLVM_PASS_TIMINGS', False):
            foo(10)

        md = foo.get_metadata(foo.signatures[0])
        timings = md['llvm_pass_timings']
        # Check that the right message is returned
        self.assertEqual(timings.summary(), "No pass timings were recorded")
        # Check that None is returned
        self.assertIsNone(timings.get_total_time())
        # Check that empty list is returned
        self.assertEqual(timings.list_longest_first(), [])


if __name__ == "__main__":
    unittest.main()
