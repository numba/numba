
import numba.unittest_support as unittest

import argparse
import collections
import functools
import gc
import sys
import warnings
from unittest import result, runner

from numba.utils import PYVERSION


# "unittest.main" is really the TestProgram class!
# (defined in a module named itself "unittest.main"...)

class NumbaTestProgram(unittest.main):
    """
    A TestProgram subclass adding a -R option to enable reference leak
    testing.
    """

    refleak = False

    def __init__(self, *args, **kwargs):
        self.discovered_suite = kwargs.pop('suite', None)
        super(NumbaTestProgram, self).__init__(*args, **kwargs)

    def createTests(self):
        if self.discovered_suite is not None:
            self.test = self.discovered_suite
        else:
            super(NumbaTestProgram, self).createTests()

    def _getParentArgParser(self):
        # NOTE: this hook only exists on Python 3.4+. The option won't be
        # added in earlier versions.
        parser = super(NumbaTestProgram, self)._getParentArgParser()
        if self.testRunner is None:
            parser.add_argument('-R', '--refleak', dest='refleak',
                                action='store_true',
                                help='Detect reference / memory leaks')
        return parser

    def runTests(self):
        if self.refleak:
            self.testRunner = RefleakTestRunner
            if not hasattr(sys, "gettotalrefcount"):
                warnings.warn("detecting reference leaks requires a debug build "
                              "of Python, only memory leaks will be detected")
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
        func2 = lambda: 42

    # Flush standard output, so that buffered data is sent to the OS and
    # associated Python objects are reclaimed.
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()

    sys._clear_type_cache()
    # This also clears the various internal CPython freelists.
    gc.collect()
    return func1(), func2()


class ReferenceLeakError(RuntimeError):
    pass


class IntPool(collections.defaultdict):

    def __missing__(self, key):
        return key


class RefleakTestResult(runner.TextTestResult):

    warmup = 3
    repetitions = 6

    def _huntLeaks(self, test):
        sys.stderr.flush()

        repcount = self.repetitions
        nwarmup = self.warmup
        rc_deltas = [0] * repcount
        alloc_deltas = [0] * repcount
        # Preallocate ints likely to be stored in rc_deltas and alloc_deltas,
        # to make sys.getallocatedblocks() less flaky.
        _int_pool = IntPool()
        for i in range(-200, 200):
            _int_pool[i]

        for i in range(repcount):
            # Use a pristine, silent result object to avoid recursion
            res = result.TestResult()
            test.run(res)
            # Poorly-written tests may fail when run several times.
            # In this case, abort the refleak run and report the failure.
            if not res.wasSuccessful():
                self.failures.extend(res.failures)
                self.errors.extend(res.errors)
                raise AssertionError
            del res
            alloc_after, rc_after = _refleak_cleanup()
            if i >= nwarmup:
                rc_deltas[i] = _int_pool[rc_after - rc_before]
                alloc_deltas[i] = _int_pool[alloc_after - alloc_before]
            alloc_before, rc_before = alloc_after, rc_after
        return rc_deltas, alloc_deltas

    def addSuccess(self, test):
        try:
            rc_deltas, alloc_deltas = self._huntLeaks(test)
        except AssertionError:
            # Test failed when repeated
            assert not self.wasSuccessful()
            return

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
