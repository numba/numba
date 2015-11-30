from __future__ import print_function, division, absolute_import

import sys
import os
import functools
import argparse
import collections
import contextlib
import cProfile
import multiprocessing
from fnmatch import fnmatch
from os.path import dirname, join, abspath, relpath

import numba.unittest_support as unittest
from unittest import result, runner, signals, suite, loader, case
from numba.utils import PYVERSION, StringIO
from numba import config

try:
    from multiprocessing import TimeoutError
except ImportError:
    from Queue import Empty as TimeoutError


class TestLister(object):
    """Simply list available tests rather than running them."""
    def __init__(self):
        pass

    def run(self, test):
        result = runner.TextTestResult(sys.stderr, descriptions=True, verbosity=1)
        self._test_list = _flatten_suite(test)
        for t in self._test_list:
            print(t.id())
        print('%d tests found'%len(self._test_list))
        return result
        
class SerialSuite(unittest.TestSuite):
    """A simple marker to make sure tests in this suite are run serially."""
    
    pass

def load_testsuite(loader, dir):
    """Find tests in 'dir'."""
    
    suite = unittest.TestSuite()
    files=[]
    for f in os.listdir(dir):
        path=join(dir, f)
        if os.path.isfile(path) and fnmatch(f, 'test_*.py'):
            files.append(f)
        elif os.path.isfile(join(path, '__init__.py')):
            suite.addTests(loader.discover(path))
    for f in files:
        # turn 'f' into a filename relative to the toplevel dir...
        f = relpath(join(dir, f), loader._top_level_dir)
        # ...and translate it to a module name.
        f = os.path.splitext(os.path.normpath(f.replace(os.path.sep, '.')))[0]
        suite.addTests(loader.loadTestsFromName(f))
    return suite

class TestLoader(loader.TestLoader):

    _top_level_dir=dirname(dirname(__file__))
    
    def discover(self, start_dir, pattern='test*.py', top_level_dir=None):
        """Upstream discover doesn't consider top-level 'load_tests' functions.
        If we find load_tests in start_dir, deal with it here. Otherwise
        forward the call to the base class."""
        top = top_level_dir or self._top_level_dir
        location = top and join(abspath(top), start_dir) or start_dir
        if os.path.isfile(join(location, '__init__.py')):
            name = self._get_name_from_path(location)
            try:
                package = self._get_module_from_name(name)
                load_tests = getattr(package, 'load_tests', None)
                if load_tests:
                    return load_tests(self, [], pattern)
            except case.SkipTest as e:
                return _make_skipped_test(name, e, self.suiteClass)
            else:
                return super(TestLoader, self).discover(start_dir, pattern, top)
        else:
            try:
                __import__(start_dir)
                module = sys.modules[start_dir]
                load_tests = getattr(module, 'load_tests', None)
                if load_tests:
                    return load_tests(self, [], pattern)
            except:
                pass
            return super(TestLoader, self).discover(start_dir, pattern, top)
            


# "unittest.main" is really the TestProgram class!
# (defined in a module named itself "unittest.main"...)

class NumbaTestProgram(unittest.main):
    """
    A TestProgram subclass adding the following options:
    * a -R option to enable reference leak detection
    * a --profile option to enable profiling of the test run
    * a -m option for parallel execution
    * a -l option to (only) list tests

    Currently the options are only added in 3.4+.
    """

    refleak = False
    profile = False
    multiprocess = False
    list = False
    
    def __init__(self, *args, **kwargs):
        # Disable interpreter fallback if we are running the test suite
        if config.COMPATIBILITY_MODE:
            warnings.warn("Unset INTERPRETER_FALLBACK")
            config.COMPATIBILITY_MODE = False

        # The default test loader is buggy in its handling of load_tests().
        # See http://bugs.python.org/issue25520
        kwargs['testLoader'] = TestLoader()

        # HACK to force unittest not to change warning display options
        # (so that NumbaWarnings don't appear all over the place)
        sys.warnoptions.append(':x')
        self.nomultiproc = kwargs.pop('nomultiproc', False)
        super(NumbaTestProgram, self).__init__(*args, **kwargs)

    def _getParentArgParser(self):
        # NOTE: this hook only exists on Python 3.4+. The options won't be
        # added in earlier versions (which use optparse - 3.3 - or getopt()
        # - 2.x).
        parser = super(NumbaTestProgram, self)._getParentArgParser()
        if self.testRunner is None:
            parser.add_argument('-R', '--refleak', dest='refleak',
                                action='store_true',
                                help='Detect reference / memory leaks')
        parser.add_argument('-m', '--multiprocess', dest='multiprocess',
                            action='store_true',
                            help='Parallelize tests')
        parser.add_argument('-l', '--list', dest='list',
                            action='store_true',
                            help='List tests without running them')
        parser.add_argument('--profile', dest='profile',
                            action='store_true',
                            help='Profile the test run')
        return parser

    def parseArgs(self, argv):
        if '-l' in argv:
            argv.remove('-l')
            self.list = True
        if PYVERSION < (3, 4) and '-m' in argv:
            # We want '-m' to work on all versions, emulate this option.
            argv.remove('-m')
            self.multiprocess = True
        super(NumbaTestProgram, self).parseArgs(argv)
        # in Python 2.7, the 'discover' option isn't implicit.
        # So if no test names were provided and 'self.test' is empty,
        # we assume discovery hasn't been done yet.
        if (not getattr(self, 'testNames', None) and
            not self.test or (isinstance(self.test, suite.BaseTestSuite) and
                              not self.test.countTestCases())):
            self._do_discovery([])
        if self.verbosity <= 0:
            # We aren't interested in informational messages / warnings when
            # running with '-q'.
            self.buffer = True
            
    def _do_discovery(self, argv, Loader=None):
        """Upstream _do_discovery doesn't find our load_tests() functions."""

        loader = TestLoader() if Loader is None else Loader()
        topdir = abspath(dirname(dirname(__file__)))
        tests = loader.discover(join(topdir, 'numba/tests'), '*.py', topdir)
        self.test = SerialSuite(tests)
        
    def runTests(self):
        if self.refleak:
            self.testRunner = RefleakTestRunner

            if not hasattr(sys, "gettotalrefcount"):
                warnings.warn("detecting reference leaks requires a debug build "
                              "of Python, only memory leaks will be detected")

        elif self.list:
            self.testRunner = TestLister()

        elif self.testRunner is None:
            self.testRunner = unittest.TextTestRunner

        if self.multiprocess and not self.nomultiproc:
            self.testRunner = ParallelTestRunner(self.testRunner,
                                                 verbosity=self.verbosity,
                                                 failfast=self.failfast,
                                                 buffer=self.buffer)

        def run_tests_real():
            super(NumbaTestProgram, self).runTests()

        if self.profile:
            filename = os.path.splitext(
                os.path.basename(sys.modules['__main__'].__file__)
                )[0] + '.prof'
            p = cProfile.Profile(timer=time.perf_counter)  # 3.3+
            p.enable()
            try:
                p.runcall(run_tests_real)
            finally:
                p.disable()
                print("Writing test profile data into %r" % (filename,))
                p.dump_stats(filename)
        else:
            run_tests_real()


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
        self.stream.flush()

        repcount = self.repetitions
        nwarmup = self.warmup
        rc_deltas = [0] * (repcount - nwarmup)
        alloc_deltas = [0] * (repcount - nwarmup)
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
                rc_deltas[i - nwarmup] = _int_pool[rc_after - rc_before]
                alloc_deltas[i - nwarmup] = _int_pool[alloc_after - alloc_before]
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
                    test, deltas, item_name, sum(deltas))
                failed = True
                try:
                    raise ReferenceLeakError(msg)
                except Exception:
                    exc_info = sys.exc_info()
                if self.showAll:
                    self.stream.write("%s = %r " % (item_name, deltas))
                self.addFailure(test, exc_info)

        if not failed:
            super(RefleakTestResult, self).addSuccess(test)


class RefleakTestRunner(runner.TextTestRunner):
    resultclass = RefleakTestResult


def _flatten_suite(test):
    """Expand suite into list of tests
    """
    if isinstance(test, unittest.TestSuite):
        tests = []
        for x in test:
            tests.extend(_flatten_suite(x))
        return tests
    else:
        return [test]

class ParallelTestResult(runner.TextTestResult):
    """
    A TestResult able to inject results from other results.
    """

    def add_results(self, result):
        """
        Add the results from the other *result* to this result.
        """
        self.stream.write(result.stream.getvalue())
        self.stream.flush()
        self.testsRun += result.testsRun
        self.failures.extend(result.failures)
        self.errors.extend(result.errors)
        self.skipped.extend(result.skipped)
        self.expectedFailures.extend(result.expectedFailures)
        self.unexpectedSuccesses.extend(result.unexpectedSuccesses)


class _MinimalResult(object):
    """
    A minimal, picklable TestResult-alike object.
    """

    __slots__ = (
        'failures', 'errors', 'skipped', 'expectedFailures',
        'unexpectedSuccesses', 'stream', 'shouldStop', 'testsRun',
        'test_id')

    def fixup_case(self, case):
        """
        Remove any unpicklable attributes from TestCase instance *case*.
        """
        # Python 3.3 doesn't reset this one.
        case._outcomeForDoCleanups = None

    def __init__(self, original_result, test_id=None):
        for attr in self.__slots__:
            setattr(self, attr, getattr(original_result, attr, None))
        for case, _ in self.expectedFailures:
            self.fixup_case(case)
        for case, _ in self.errors:
            self.fixup_case(case)
        for case, _ in self.failures:
            self.fixup_case(case)
        self.test_id = test_id


class _FakeStringIO(object):
    """
    A trivial picklable StringIO-alike for Python 2.
    """

    def __init__(self, value):
        self._value = value

    def getvalue(self):
        return self._value


class _MinimalRunner(object):
    """
    A minimal picklable object able to instantiate a runner in a
    child process and run a test case with it.
    """

    def __init__(self, runner_cls, runner_args):
        self.runner_cls = runner_cls
        self.runner_args = runner_args

    # Python 2 doesn't know how to pickle instance methods, so we use __call__
    # instead.

    def __call__(self, test):
        # Executed in child process
        kwargs = self.runner_args
        # Force recording of output in a buffer (it will be printed out
        # by the parent).
        kwargs['stream'] = StringIO()
        runner = self.runner_cls(**kwargs)
        result = runner._makeResult()
        # Avoid child tracebacks when Ctrl-C is pressed.
        signals.installHandler()
        signals.registerResult(result)
        result.failfast = runner.failfast
        result.buffer = runner.buffer
        with self.cleanup_object(test):
            test(result)
        # HACK as cStringIO.StringIO isn't picklable in 2.x
        result.stream = _FakeStringIO(result.stream.getvalue())
        return _MinimalResult(result, test.id())

    @contextlib.contextmanager
    def cleanup_object(self, test):
        """
        A context manager which cleans up unwanted attributes on a test case
        (or any other object).
        """
        vanilla_attrs = set(test.__dict__)
        try:
            yield test
        finally:
            spurious_attrs = set(test.__dict__) - vanilla_attrs
            for name in spurious_attrs:
                del test.__dict__[name]


def _split_nonparallel_tests(test):
    """split test suite into parallel and serial tests."""
    ptests = []
    stests = []
    if isinstance(test, SerialSuite):
        stests.extend(_flatten_suite(test))
    elif isinstance(test, unittest.TestSuite):
        for t in test:
            p, s = _split_nonparallel_tests(t)
            ptests.extend(p)
            stests.extend(s)
    else:
        ptests = [test]
    return ptests, stests

class ParallelTestRunner(runner.TextTestRunner):
    """
    A test runner which delegates the actual running to a pool of child
    processes.
    """

    resultclass = ParallelTestResult
    # A test can't run longer than 2 minutes
    timeout = 120

    def __init__(self, runner_cls, **kwargs):
        runner.TextTestRunner.__init__(self, **kwargs)
        self.runner_cls = runner_cls
        self.runner_args = kwargs

    def _run_inner(self, result):
        # We hijack TextTestRunner.run()'s inner logic by passing this
        # method as if it were a test case.
        child_runner = _MinimalRunner(self.runner_cls, self.runner_args)
        pool = multiprocessing.Pool()

        try:
            self._run_parallel_tests(result, pool, child_runner)
        finally:
            # Kill the still active workers
            pool.terminate()
            pool.join()
        stests = SerialSuite(self._stests)
        stests.run(result)
        return result

    def _run_parallel_tests(self, result, pool, child_runner):
        remaining_ids = set(t.id() for t in self._ptests)
        it = pool.imap_unordered(child_runner, self._ptests)
        while True:
            try:
                child_result = it.__next__(self.timeout)
            except StopIteration:
                return
            except TimeoutError as e:
                # Diagnose the names of unfinished tests
                msg = ("%s [unfinished tests: %s]"
                       % (str(e), ", ".join(map(repr, sorted(remaining_ids))))
                       )
                e.args = (msg,) + e.args[1:]
                raise e
            result.add_results(child_result)
            remaining_ids.discard(child_result.test_id)
            if child_result.shouldStop:
                return

    def run(self, test):
        self._ptests, self._stests = _split_nonparallel_tests(test)
        # This will call self._run_inner() on the created result object,
        # and print out the detailed test results at the end.
        return super(ParallelTestRunner, self).run(self._run_inner)


def allow_interpreter_mode(fn):
    """Temporarily re-enable intepreter mode
    """
    @functools.wraps(fn)
    def _core(*args, **kws):
        config.COMPATIBILITY_MODE = True
        try:
            fn(*args, **kws)
        finally:
            config.COMPATIBILITY_MODE = False
    return _core


def run_tests(argv=None, xmloutput=None, verbosity=1, nomultiproc=False):
    """
    args
    ----
    - xmloutput [str or None]
        Path of XML output directory (optional)
    - verbosity [int]
        Verbosity level of tests output

    Returns the TestResult object after running the test *suite*.
    """

    if xmloutput is not None:
        import xmlrunner
        runner = xmlrunner.XMLTestRunner(output=xmloutput)
    else:
        runner = None
    prog = NumbaTestProgram(argv=argv,
                            module=None,
                            testRunner=runner, exit=False,
                            verbosity=verbosity,
                            nomultiproc=nomultiproc)
    return prog.result


def test(*args, **kwargs):

    return run_tests(argv=['<main>'] + list(args), **kwargs).wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run_tests(sys.argv) else 1)
