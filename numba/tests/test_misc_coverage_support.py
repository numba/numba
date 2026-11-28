import threading
import unittest
from unittest.mock import MagicMock, patch

from numba.tests.support import TestCase

from numba import njit
from numba.core import ir
from numba.misc.coverage_support import (
    NotifyLocBase,
    _the_registry,
    coverage_available,
)


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


class TestNumbaTracerToolId(TestCase):
    @unittest.skipUnless(coverage_available, "coverage not installed")
    def test_tool_id_in_tracer_kwargs_accepted(self):
        # Regression test for https://github.com/numba/numba/issues/10589.
        # coverage >= 7.13.1 injects tool_id into tracer_kwargs when using the
        # sysmon core; NumbaTracer must accept it without raising TypeError.
        from numba.misc.coverage_support import NotifyCompilerCoverage

        collector = MagicMock()
        collector.core.tracer_kwargs = {'tool_id': 1, 'packed_arcs': False}
        collector.branch = False
        collector.should_start_context = None
        collector.switch_context = None
        collector.tracers = []

        NotifyCompilerCoverage(collector)


@unittest.skipUnless(coverage_available, "coverage not installed")
class TestNumbaTracerLockData(TestCase):
    # Regression tests for coverage >= 7.17.5, which removed
    # Collector.lock_data and Collector.unlock_data.

    def make_collector(self, data_lock):
        collector = MagicMock()
        collector.core.tracer_kwargs = {'packed_arcs': False}
        collector.branch = False
        collector.should_start_context = None
        collector.switch_context = None
        collector.tracers = []
        collector.data_lock = data_lock
        del collector.lock_data
        del collector.unlock_data
        return collector

    def test_data_lock_methods_used(self):
        from numba.misc.coverage_support import NotifyCompilerCoverage

        lock = threading.Lock()
        tracer = NotifyCompilerCoverage(self.make_collector(lock))._tracer

        tracer.lock_data()
        self.assertTrue(lock.locked())
        tracer.unlock_data()
        self.assertFalse(lock.locked())

    def test_no_data_lock(self):
        from numba.misc.coverage_support import NotifyCompilerCoverage

        tracer = NotifyCompilerCoverage(self.make_collector(None))._tracer

        tracer.lock_data()
        tracer.unlock_data()

    @TestCase.run_test_in_subprocess(envvars={"NUMBA_JIT_COVERAGE": "1"})
    def test_compiled_lines_recorded(self):
        import coverage

        # Ignore any coverage configuration (e.g. `source`, `branch`) and
        # measure this file even when Numba is installed in site-packages.
        cov = coverage.Coverage(
            data_file=None, config_file=False, include=[__file__], branch=False
        )
        cov.start()
        try:
            @njit
            def foo():
                return 123

            res = foo()
        finally:
            cov.stop()

        self.assertEqual(res, 123)
        # The body only runs as compiled code, so it is recorded only if
        # Numba's tracer reported it.
        body = foo.__code__.co_firstlineno + 2
        self.assertIn(body, cov.get_data().lines(__file__) or [])


if __name__ == "__main__":
    unittest.main()
