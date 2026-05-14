import unittest
from unittest.mock import patch

from numba.tests.support import TestCase

from numba import njit
from numba.core import ir
from numba.misc.coverage_support import NotifyLocBase, _the_registry


class TestMiscCoverageSupport(TestCase):
    @TestCase.run_test_in_subprocess(envvars={"NUMBA_JIT_COVERAGE": "1"})
    def test_custom_loc_notifier(self):
        class MyNotify(NotifyLocBase):
            records = []

            def notify(self, loc):
                self.records.append(("NOTIFY", loc))

            def close(self):
                self.records.append(("CLOSE", None))

        def next_col(code):
            last_line = None
            if (hasattr(code, "co_positions")):
                for pos in code.co_positions():
                    if (last_line is None or pos[0] != last_line):
                        last_line = pos[0]
                        yield pos[2]

        # Patch to install registry for testing
        new_the_registry = _the_registry + [MyNotify]
        gv = "numba.misc.coverage_support._the_registry"
        with patch(gv, new_the_registry):

            @njit
            def foo():
                return 123

            res = foo()

        self.assertEqual(res, 123)

        # offset by +2 because:
        # +1 for the decorator
        # +1 for the `def` line
        first_offset = 2
        offset = foo.__code__.co_firstlineno + first_offset
        col_offsets = list(next_col(foo.__code__))
        if (len(col_offsets) == 0 or col_offsets is None):
            col_offsets = [None]
        loc = ir.Loc(__file__, 1, col_offsets[-1])
        self.assertIn(("NOTIFY", loc.with_lineno(offset, col_offsets[-1])),
                      MyNotify.records)
        self.assertIn(("CLOSE", None), MyNotify.records)

        # Test dead branch pruned
        with patch(gv, new_the_registry):
            cond = False

            @njit
            def foo():
                if cond:
                    return 321
                return 123

            res = foo()

        self.assertEqual(res, 123)

        # `if cond` line is compiled
        offset = foo.__code__.co_firstlineno + first_offset
        col_offsets = list(next_col(foo.__code__))
        if (len(col_offsets) == 0 or col_offsets is None):
            col_offsets = [None, None, None]
        self.assertIn(("NOTIFY", loc.with_lineno(offset, col_offsets[2])),
                      MyNotify.records)

        # `    return 321` line is not compiled
        self.assertNotIn(
            ("NOTIFY", loc.with_lineno(offset + 1, col_offsets[-1])),
            MyNotify.records
        )

        # `    return 123` line is compiled
        self.assertIn(("NOTIFY", loc.with_lineno(offset + 2, col_offsets[-1])),
                      MyNotify.records)

        self.assertIn(("CLOSE", None), MyNotify.records)


if __name__ == "__main__":
    unittest.main()
