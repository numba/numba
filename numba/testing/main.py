from __future__ import print_function, division, absolute_import

import numba.unittest_support as unittest

import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import warnings

from unittest import result, runner, signals, suite, loader, case

from .loader import TestLoader
from numba.utils import PYVERSION, StringIO
from numba import config

try:
    from multiprocessing import TimeoutError
except ImportError:
    from Queue import Empty as TimeoutError


def make_tag_decorator(known_tags):
    """
    Create a decorator allowing tests to be tagged with the *known_tags*.
    """

    def tag(*tags):
        """
        Tag a test method with the given tags.
        Can be used in conjunction with the --tags command-line argument
        for runtests.py.
        """
        for t in tags:
            if t not in known_tags:
                raise ValueError("unknown tag: %r" % (t,))

        def decorate(func):
            if (not callable(func) or isinstance(func, type)
                or not func.__name__.startswith('test_')):
                raise TypeError("@tag(...) should be used on test methods")
            try:
                s = func.tags
            except AttributeError:
                s = func.tags = set()
            s.update(tags)
            return func
        return decorate

    return tag


def test_mtime(x):
    return str(os.path.getmtime(inspect.getfile(x.__class__))) + str(x)


def parse_slice(useslice):
    """Parses the argument string "useslice" as the arguments to the `slice()`
    constructor and returns a slice object that's been instantiated with those
    arguments. i.e. input useslice="1,20,2" leads to output `slice(1, 20, 2)`.
    """
    try:
        l = {}
        exec("sl = slice(%s)" % useslice, l)
        return l['sl']
    except Exception:
        msg = ("Expected arguments consumable by 'slice' to follow "
                "option `-j`, found '%s'" % useslice)
        raise ValueError(msg)


class TestLister(object):
    """Simply list available tests rather than running them."""
    def __init__(self, useslice):
        self.useslice = parse_slice(useslice)

    def run(self, test):
        result = runner.TextTestResult(sys.stderr, descriptions=True, verbosity=1)
        self._test_list = _flatten_suite(test)
        masked_list = self._test_list[self.useslice]
        self._test_list.sort(key=test_mtime)
        for t in masked_list:
            print(t.id())
        print('%d tests found. %s selected' % (len(self._test_list), len(masked_list)))
        return result


class SerialSuite(unittest.TestSuite):
    """A simple marker to make sure tests in this suite are run serially.

    Note: As the suite is going through internals of unittest,
          it may get unpacked and stuffed into a plain TestSuite.
          We need to set an attribute on the TestCase objects to
          remember they should not be run in parallel.
    """

    def addTest(self, test):
        if not isinstance(test, unittest.TestCase):
            # It's a sub-suite, recurse
            for t in test:
                self.addTest(t)
        else:
            # It's a test case, mark it serial
            test._numba_parallel_test_ = False
            super(SerialSuite, self).addTest(test)


class BasicTestRunner(runner.TextTestRunner):
    def __init__(self, useslice, **kwargs):
        runner.TextTestRunner.__init__(self, **kwargs)
        self.useslice = parse_slice(useslice)

    def run(self, test):
        run = _flatten_suite(test)[self.useslice]
        run.sort(key=test_mtime)
        wrapped = unittest.TestSuite(run)
        return super(BasicTestRunner, self).run(wrapped)


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
    useslice = None
    list = False
    tags = None
    exclude_tags = None
    random_select = None
    random_seed = 42

    def __init__(self, *args, **kwargs):
        # Disable interpreter fallback if we are running the test suite
        if config.COMPATIBILITY_MODE:
            warnings.warn("Unset INTERPRETER_FALLBACK")
            config.COMPATIBILITY_MODE = False

        topleveldir = kwargs.pop('topleveldir', None)
        kwargs['testLoader'] = TestLoader(topleveldir)

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
                            nargs='?',
                            type=int,
                            const=multiprocessing.cpu_count(),
                            help='Parallelize tests')
        parser.add_argument('-l', '--list', dest='list',
                            action='store_true',
                            help='List tests without running them')
        parser.add_argument('--tags', dest='tags', type=str,
                            help='Comma-separated list of tags to select '
                                 'a subset of the test suite')
        parser.add_argument('--exclude-tags', dest='exclude_tags', type=str,
                            help='Comma-separated list of tags to de-select '
                                 'a subset of the test suite')
        parser.add_argument('--random', dest='random_select', type=float,
                            help='Random proportion of tests to select')
        parser.add_argument('--profile', dest='profile',
                            action='store_true',
                            help='Profile the test run')
        parser.add_argument('-j', '--slice', dest='useslice', nargs='?',
                            type=str, const="None",
                            help='Slice the test sequence')
        parser.add_argument('-g', '--gitdiff', dest='gitdiff',
                            action='store_true',
                            help=('Run tests from changes made against'
                                  'origin/master as identified by `git diff`'))
        return parser

    def _handle_tags(self, argv, tagstr):
        found = None
        for x in argv:
            if tagstr in x:
                if found is None:
                    found = x
                else:
                    raise ValueError("argument %s supplied repeatedly" % tagstr)

        if found is not None:
            posn = argv.index(found)
            try:
                if found == tagstr: # --tagstr <arg>
                    tag_args = argv[posn + 1].strip()
                    argv.remove(tag_args)
                else: # --tagstr=<arg>
                    if '=' in found:
                        tag_args =  found.split('=')[1].strip()
                    else:
                        raise AssertionError('unreachable')
            except IndexError:
                # at end of arg list, raise
                msg = "%s requires at least one tag to be specified"
                raise ValueError(msg % tagstr)
            # see if next arg is "end options" or some other flag
            if tag_args.startswith('-'):
                raise ValueError("tag starts with '-', probably a syntax error")
            # see if tag is something like "=<tagname>" which is likely a syntax
            # error of form `--tags =<tagname>`, note the space prior to `=`.
            if '=' in tag_args:
                msg = "%s argument contains '=', probably a syntax error"
                raise ValueError(msg % tagstr)
            attr = tagstr[2:].replace('-', '_')
            setattr(self, attr, tag_args)
            argv.remove(found)


    def parseArgs(self, argv):
        if '-l' in argv:
            argv.remove('-l')
            self.list = True
        if PYVERSION < (3, 4):
            if '-m' in argv:
                # We want '-m' to work on all versions, emulate this option.
                dashm_posn = argv.index('-m')
                # the default number of processes to use
                nprocs = multiprocessing.cpu_count()
                # see what else is in argv
                # ensure next position is safe for access
                try:
                    m_option = argv[dashm_posn + 1]
                    # see if next arg is "end options"
                    if m_option != '--':
                        #try and parse the next arg as an int
                        try:
                            nprocs = int(m_option)
                        except Exception:
                            msg = ('Expected an integer argument to '
                                'option `-m`, found "%s"')
                            raise ValueError(msg % m_option)
                        # remove the value of the option
                        argv.remove(m_option)
                    # else end options, use defaults
                except IndexError:
                    # at end of arg list, use defaults
                    pass

                self.multiprocess = nprocs
                argv.remove('-m')

            if '-j' in argv:
                # We want '-s' to work on all versions, emulate this option.
                dashs_posn = argv.index('-j')
                j_option = argv[dashs_posn + 1]
                self.useslice = j_option
                argv.remove(j_option)
                argv.remove('-j')

            self.gitdiff = False
            if '-g' in argv:
                self.gitdiff = True
                argv.remove('-g')

            # handle tags
            self._handle_tags(argv, '--tags')
            self._handle_tags(argv, '--exclude-tags')

        super(NumbaTestProgram, self).parseArgs(argv)

        # If at this point self.test doesn't exist, it is because
        # no test ID was given in argv. Use the default instead.
        if not hasattr(self, 'test') or not self.test.countTestCases():
            self.testNames = (self.defaultTest,)
            self.createTests()

        if self.tags:
            tags = [s.strip() for s in self.tags.split(',')]
            self.test = _choose_tagged_tests(self.test, tags, mode='include')

        if self.exclude_tags:
            tags = [s.strip() for s in self.exclude_tags.split(',')]
            self.test = _choose_tagged_tests(self.test, tags, mode='exclude')

        if self.random_select:
            self.test = _choose_random_tests(self.test, self.random_select,
                                             self.random_seed)

        if self.gitdiff:
            self.test = _choose_gitdiff_tests(self.test)

        if self.verbosity <= 0:
            # We aren't interested in informational messages / warnings when
            # running with '-q'.
            self.buffer = True

    def _do_discovery(self, argv, Loader=None):
        # Disable unittest's implicit test discovery when parsing
        # CLI arguments, as it can select other tests than Numba's
        # (e.g. some test_xxx module that may happen to be directly
        #  reachable from sys.path)
        return

    def runTests(self):
        if self.refleak:
            self.testRunner = RefleakTestRunner

            if not hasattr(sys, "gettotalrefcount"):
                warnings.warn("detecting reference leaks requires a debug build "
                              "of Python, only memory leaks will be detected")

        elif self.list:
            self.testRunner = TestLister(self.useslice)

        elif self.testRunner is None:
            self.testRunner = BasicTestRunner(self.useslice,
                                              verbosity=self.verbosity,
                                              failfast=self.failfast,
                                              buffer=self.buffer)

        if self.multiprocess and not self.nomultiproc:
            if self.multiprocess < 1:
                msg = ("Value specified for the number of processes to use in "
                    "running the suite must be > 0")
                raise ValueError(msg)
            self.testRunner = ParallelTestRunner(runner.TextTestRunner,
                                                 self.multiprocess,
                                                 self.useslice,
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


def _flatten_suite(test):
    """
    Expand nested suite into list of test cases.
    """
    if isinstance(test, (unittest.TestSuite, list, tuple)):
        tests = []
        for x in test:
            tests.extend(_flatten_suite(x))
        return tests
    else:
        return [test]

def _choose_gitdiff_tests(tests):
    try:
        from git import Repo
    except ImportError:
        raise ValueError("gitpython needed for git functionality")
    repo = Repo('.')
    path = os.path.join('numba', 'tests')
    target = 'origin/master..HEAD'
    gdiff_paths = repo.git.diff(target, path, name_only=True).split()
    # normalise the paths as they are unix style from repo.git.diff
    gdiff_paths = [os.path.normpath(x) for x in gdiff_paths]
    selected = []
    if PYVERSION > (2, 7): # inspect output changes in py3
        gdiff_paths = [os.path.join(repo.working_dir, x) for x in gdiff_paths]
    for test in _flatten_suite(tests):
        assert isinstance(test, unittest.TestCase)
        fname = inspect.getsourcefile(test.__class__)
        if fname in gdiff_paths:
            selected.append(test)
    print("Git diff identified %s tests" % len(selected))
    return unittest.TestSuite(selected)

def _choose_tagged_tests(tests, tags, mode='include'):
    """
    Select tests that are tagged/not tagged with at least one of the given tags.
    Set mode to 'include' to include the tests with tags, or 'exclude' to
    exclude the tests with the tags.
    """
    selected = []
    tags = set(tags)
    for test in _flatten_suite(tests):
        assert isinstance(test, unittest.TestCase)
        func = getattr(test, test._testMethodName)
        try:
            # Look up the method's underlying function (Python 2)
            func = func.im_func
        except AttributeError:
            pass

        found_tags = getattr(func, 'tags', None)
        # only include the test if the tags *are* present
        if mode == 'include':
            if found_tags is not None and found_tags & tags:
                selected.append(test)
        elif mode == 'exclude':
            # only include the test if the tags *are not* present
            if found_tags is None or not (found_tags & tags):
                selected.append(test)
        else:
            raise ValueError("Invalid 'mode' supplied: %s." % mode)
    return unittest.TestSuite(selected)


def _choose_random_tests(tests, ratio, seed):
    """
    Choose a given proportion of tests at random.
    """
    rnd = random.Random()
    rnd.seed(seed)
    if isinstance(tests, unittest.TestSuite):
        tests = _flatten_suite(tests)
    tests = rnd.sample(tests, int(len(tests) * ratio))
    tests = sorted(tests, key=lambda case: case.id())
    return unittest.TestSuite(tests)


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
            if not set(deltas) <= set((1, 0, -1)):
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


def _split_nonparallel_tests(test, sliced=slice(None)):
    """
    Split test suite into parallel and serial tests.
    """
    ptests = []
    stests = []

    flat = _flatten_suite(test)[sliced]

    def is_parallelizable_test_case(test):
        # Guard for the fake test case created by unittest when test
        # discovery fails, as it isn't picklable (e.g. "LoadTestsFailure")
        method_name = test._testMethodName
        method = getattr(test, method_name)
        if method.__name__ != method_name and method.__name__ == "testFailure":
            return False
        # Was parallel execution explicitly disabled?
        return getattr(test, "_numba_parallel_test_", True)

    for t in flat:
        if is_parallelizable_test_case(t):
            ptests.append(t)
        else:
            stests.append(t)

    return ptests, stests

# A test can't run longer than 10 minutes
_TIMEOUT = 600

class ParallelTestRunner(runner.TextTestRunner):
    """
    A test runner which delegates the actual running to a pool of child
    processes.
    """

    resultclass = ParallelTestResult
    timeout = _TIMEOUT

    def __init__(self, runner_cls, nprocs, useslice, **kwargs):
        runner.TextTestRunner.__init__(self, **kwargs)
        self.runner_cls = runner_cls
        self.nprocs = nprocs
        self.useslice = parse_slice(useslice)
        self.runner_args = kwargs

    def _run_inner(self, result):
        # We hijack TextTestRunner.run()'s inner logic by passing this
        # method as if it were a test case.
        child_runner = _MinimalRunner(self.runner_cls, self.runner_args)

        # Split the tests and recycle the worker process to tame memory usage.
        chunk_size = 100
        splitted_tests = [self._ptests[i:i + chunk_size]
                          for i in range(0, len(self._ptests), chunk_size)]

        for tests in splitted_tests:
            pool = multiprocessing.Pool(self.nprocs)
            try:
                self._run_parallel_tests(result, pool, child_runner, tests)
            except:
                # On exception, kill still active workers immediately
                pool.terminate()
                # Make sure exception is reported and not ignored
                raise
            else:
                # Close the pool cleanly unless asked to early out
                if result.shouldStop:
                    pool.terminate()
                    break
                else:
                    pool.close()
            finally:
                # Always join the pool (this is necessary for coverage.py)
                pool.join()
        if not result.shouldStop:
            stests = SerialSuite(self._stests)
            stests.run(result)
            return result

    def _run_parallel_tests(self, result, pool, child_runner, tests):
        remaining_ids = set(t.id() for t in tests)
        tests.sort(key=test_mtime)
        it = pool.imap_unordered(child_runner, tests)
        while True:
            try:
                child_result = it.__next__(self.timeout)
            except StopIteration:
                return
            except TimeoutError as e:
                # Diagnose the names of unfinished tests
                msg = ("Tests didn't finish before timeout (or crashed):\n%s"
                       % "".join("- %r\n" % tid for tid in sorted(remaining_ids))
                       )
                e.args = (msg,) + e.args[1:]
                raise e
            else:
                result.add_results(child_result)
                remaining_ids.discard(child_result.test_id)
                if child_result.shouldStop:
                    result.shouldStop = True
                    return

    def run(self, test):
        self._ptests, self._stests = _split_nonparallel_tests(test,
                                                              sliced=
                                                              self.useslice)
        print("Parallel: %s. Serial: %s" % (len(self._ptests),
                                            len(self._stests)))
        # This will call self._run_inner() on the created result object,
        # and print out the detailed test results at the end.
        return super(ParallelTestRunner, self).run(self._run_inner)
