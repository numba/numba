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


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
