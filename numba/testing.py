from __future__ import print_function, division, absolute_import
import sys
import contextlib

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO


def discover_tests(startdir):
    """Discover test under a directory
    """
    import numba.unittest_support as unittest

    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite


def run_tests(suite, descriptions=True, verbosity=2, buffer=True,
              failfast=False, xmloutput=None):
    """
    args
    ----
    - descriptions
    - verbosity
    - buffer
    - failfast
    - xmloutput [str]
        Path of XML output directory
    """
    import numba.unittest_support as unittest
    if xmloutput is not None:
        import xmlrunner
        runner = xmlrunner.XMLTestRunner(output=xmloutput)
    else:
        runner = unittest.TextTestRunner(descriptions=descriptions,
                                         verbosity=verbosity,
                                         buffer=buffer, failfast=failfast)
    result = runner.run(suite)
    return result


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
    suite = discover_tests("numba.tests")
    return run_tests(suite, **kwargs).wasSuccessful()


def _flatten_suite(test):
    """Expand suite into list of tests
    """
    from numba.unittest_support import TestSuite
    if isinstance(test, TestSuite):
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
    """
    import numba.unittest_support as unittest
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
    import numba.unittest_support as unittest
    stream = StringIO()
    with contextlib.closing(stream):
        runner = unittest.TextTestRunner(descriptions=False, verbosity=3,
                                         buffer=True, stream=stream)
        result = runner.run(suite)
        return result.wasSuccessful(), stream.getvalue()
