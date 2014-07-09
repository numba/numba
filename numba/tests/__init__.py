
import numba.unittest_support as unittest

import argparse
import functools
import gc
import sys
import warnings
from unittest import result, runner

from numba.utils import PYVERSION


# "unittest.main" is really the TestProgram class!
# (defined in a module named itself "unittest.main"...)

class NumbaTestProgram(unittest.main):
    refleak = False

    def _getParentArgParser(self):
        parser = super(NumbaTestProgram, self)._getParentArgParser()
        if PYVERSION >= (3, 4):
            parser.add_argument('-R', '--refleak', dest='refleak',
                                action='store_true',
                                help='Detect reference / memory leaks')
        return parser

    def runTests(self):
        if self.refleak:
            self.testRunner = RefleakTestRunner
        super(NumbaTestProgram, self).runTests()


# Monkey-patch unittest so that individual test modules get our custom
# options for free.
unittest.main = NumbaTestProgram


# The reference leak detection code is liberally taken and adapted from
# Python's own Lib/test/regrtest.py.

def _refleak_cleanup():
    # Collect cyclic trash and read memory statistics immediately after.
    func1 = sys.getallocatedblocks
    try:
        func2 = sys.gettotalrefcount
    except AttributeError:
        warnings.warn("detecting reference leaks requires a debug build "
                      "of Python, only memory leaks will be detected")
        func2 = lambda: 42

    # Flush standard output, so that buffered data is sent to the OS and
    # associated Python objects are reclaimed.
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()

    sys._clear_type_cache()
    gc.collect()
    return func1(), func2()

def _warm_caches():
    # char cache
    s = bytes(range(256))
    for i in range(256):
        s[i:i+1]
    # unicode cache
    x = [chr(i) for i in range(256)]
    # int cache
    x = list(range(-5, 257))


class ReferenceLeakError(RuntimeError):
    pass


class RefleakTestResult(runner.TextTestResult):

    warmup = 3
    repetitions = 6

    def _huntLeaks(self, test):
        _warm_caches()
        repcount = self.repetitions
        nwarmup = self.warmup
        rc_deltas = [0] * repcount
        alloc_deltas = [0] * repcount
        sys.stderr.flush()
        for i in range(repcount):
            # Use a pristine, silent result object to avoid recursion
            res = result.TestResult()
            test.run(res)
            assert res.wasSuccessful()
            del res
            alloc_after, rc_after = _refleak_cleanup()
            if i >= nwarmup:
                rc_deltas[i] = rc_after - rc_before
                alloc_deltas[i] = alloc_after - alloc_before
            alloc_before, rc_before = alloc_after, rc_after
        return rc_deltas, alloc_deltas

    def addSuccess(self, test):
        rc_deltas, alloc_deltas = self._huntLeaks(test)

        # These checkers return False on success, True on failure
        def check_rc_deltas(deltas):
            return any(deltas)
        def check_alloc_deltas(deltas):
            # At least 1/3rd of 0s
            if 3 * deltas.count(0) < len(deltas):
                return True
            # Nothing else than 1s, 0s and -1s
            if not set(deltas) <= set((1,0,-1)):
                return True
            return False

        failed = False

        for deltas, item_name, checker in [
            (rc_deltas, 'references', check_rc_deltas),
            (alloc_deltas, 'memory blocks', check_alloc_deltas)]:
            if checker(deltas):
                msg = '%s leaked %s %s, sum=%s' % (
                    test, deltas[self.warmup:], item_name, sum(deltas))
                failed = True
                try:
                    raise ReferenceLeakError(msg)
                except Exception:
                    exc_info = sys.exc_info()
                self.addFailure(test, exc_info)

        if not failed:
            super(RefleakTestResult, self).addSuccess(test)


class RefleakTestRunner(runner.TextTestRunner):
    resultclass = RefleakTestResult


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()
