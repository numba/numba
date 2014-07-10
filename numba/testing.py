from __future__ import print_function, division, absolute_import
import sys
import contextlib

import numba.unittest_support as unittest

from numba.tests import NumbaTestProgram
from numba.utils import StringIO


def discover_tests(startdir):
    """Discover test under a directory
    """

    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite


def run_tests(suite, xmloutput=None):
    """
    args
    ----
    - suite [TestSuite]
        A suite of all tests to run
    - xmloutput [str or None]
        Path of XML output directory (optional)

    Returns the TestResult object after running the test *suite*.
    """
    if xmloutput is not None:
        import xmlrunner
        runner = xmlrunner.XMLTestRunner(output=xmloutput)
    else:
        runner = None
    prog = NumbaTestProgram(suite=suite, testRunner=runner, exit=False)
    return prog.result


def test(**kwargs):
    """
    Run all tests under ``numba.tests``.

    kwargs
    ------
    - descriptions
    - verbosity
    - buffer
    - failfast
    - xmloutput [str]
        Path of XML output directory
    """
    from numba import cuda
    suite = discover_tests("numba.tests")
    ok = run_tests(suite, **kwargs).wasSuccessful()
    if ok:
        if cuda.is_available():
            print("== Run CUDA tests ==")
            ok = cuda.test()
        else:
            print("== Skipped CUDA tests ==")

    return ok



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


def multitest():
    """
    Run tests in multiple processes.

    Use this for running all tests under ``numba.tests`` quickly.
    This isn't compatible with command-line test options such as ``--failfast``,
    etc.
    """
    import multiprocessing as mp

    loader = unittest.TestLoader()
    startdir = "numba.tests"
    suites = loader.discover(startdir)
    tests = _flatten_suite(suites)
    # Distribute tests to multiple processes
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.imap_unordered(_multiruntest, tests)

    errct = 0
    for ok, out in results:
        if not ok:
            print()
            print("=== Error ===")
            print(out)
            errct += 1
        else:
            print('.', end='')
            sys.stdout.flush()

    print()
    if errct == 0:
        print("All passed!")
        return True
    else:
        print("Error %d/%d" % (errct, len(tests)))
        return False


def _multiruntest(suite):
    stream = StringIO()
    with contextlib.closing(stream):
        runner = unittest.TextTestRunner(descriptions=False, verbosity=3,
                                         buffer=True, stream=stream)
        result = runner.run(suite)
        return result.wasSuccessful(), stream.getvalue()


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
