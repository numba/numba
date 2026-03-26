"""
Regression test for issue #5611 / PR #10482.

CFGraph._topo_order must be stack-based (not recursive) so that
functions with large control-flow graphs do not exhaust the Python call
stack.

The test is run in a subprocess so that sys.setrecursionlimit manipulation
is completely isolated from the rest of the suite.
"""

import sys
import traceback
import unittest

from numba import njit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.tests.support import TestCase
from numba.core.bytecode import FunctionIdentity, ByteCode


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _generate_large_cfg_source(count):
    """
    Return Python source for a function whose CFG has an unusually large
    ammount of nodes.

    """
    lines = ["def _large_cfg_func(x):"]
    for i in range(count):
        lines.append(f"    dep_{i} = 1 if True else 0")
        lines.append(f"    val_{i} = x if dep_{i} else 0")
    lines.append("    return x")
    return "\n".join(lines)


def _create_cfg_from_function(func):
    """Create a CFGraph from a Python function for direct testing."""
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id)
    cfa = ControlFlowAnalysis(bc)
    cfa.run()
    return cfa.graph


# Complex test function with many branches
def _complex_func(x):
    dep_0 = 1 if True else 0
    val_0 = x if dep_0 else 0
    dep_1 = 1 if True else 0
    val_1 = x if dep_1 else 0
    return x


def _measure_trivial_compile_depth():
    """
    Return the Python frame depth at CFGraph.process() during a trivial
    njit compilation.

    We monkey-patch CFGraph.process — called once per compilation — to
    walk the frame chain a single time.  This is O(depth) total, not
    O(calls × depth) like settrace, so it is fast.
    """
    depth_sample = [0]
    orig_process = CFGraph.topo_order

    def _probing_process(self):
        f = sys._getframe()
        d = 0
        while f is not None:
            d += 1
            f = f.f_back
        if d > depth_sample[0]:
            depth_sample[0] = d
        return orig_process(self)

    CFGraph.topo_order = _probing_process
    try:
        c = _create_cfg_from_function(_complex_func)
        c.topo_order()
    finally:
        CFGraph.process = orig_process   # always restore
    return depth_sample[0]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestCFGTopoOrderNonRecursive(TestCase):

    @TestCase.run_test_in_subprocess
    def test_topo_order_no_recursion_in_large_cfg(self):
        """
        Regression tests for PR #10482: _find_topo_order must not use recursive
        Python calls, so that large CFGs cannot exhaust the stack.

        Steps
        -----
        1.  Measure the frame depth at CFGraph.process() during a trivial
            njit compilation by patching the method to walk the frame
            chain once; add 100 % headroom → Bsize.
        2.  Lower sys.setrecursionlimit to Bsize.
        3.  Attempt to compile a function with 100 ternary-expression
            pairs.
        4.  Catch any RecursionError.
        5.  Restore sys.setrecursionlimit unconditionally.
        6.  Fail the test and print the traceback if a RecursionError was
            raised.
        """

        # ---- 1. Derive Bsize ------------------------------------------------
        depth_at_cfg_process = _measure_trivial_compile_depth()
        Bsize = int(depth_at_cfg_process * 2.0)

        # ---- 2. Lower the recursion limit -----------------------------------
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(Bsize)

        # ---- 3 & 4. Compile large CFG; catch any overflow -------------------
        caught_exc = None
        try:
            c = _create_cfg_from_function(_generate_large_cfg_source(100))
            c.topo_order()
        except RecursionError as exc:
            caught_exc = exc
        finally:
            # ---- 5. Restore limit unconditionally ---------------------------
            sys.setrecursionlimit(original_limit)

        # ---- 6. Verify the error site (or celebrate clean success) ----------
        if caught_exc is None:
            # The entire pipeline is iterative end-to-end: strongest result.
            return
        else:
            # Fail
            msg = (
                "Unexpected RecursionError."
                "Potential regression related to fiding toplogical order.\n\n"
                "Full traceback:\n"
                + "".join(traceback.format_tb(caught_exc.__traceback__)))
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
