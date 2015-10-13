from __future__ import print_function, division, absolute_import

import sys
import functools

from numba import config


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


def discover_tests(startdir):
    """Discover test under a directory
    """
    # Avoid importing unittest
    from numba import unittest_support as unittest
    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite


def run_tests(xmloutput=None, verbosity=1, nomultiproc=False):
    """
    args
    ----
    - xmloutput [str or None]
        Path of XML output directory (optional)
    - verbosity [int]
        Verbosity level of tests output

    Returns the TestResult object after running the test *suite*.
    """
    from numba.tests import NumbaTestProgram

    if xmloutput is not None:
        import xmlrunner
        runner = xmlrunner.XMLTestRunner(output=xmloutput)
    else:
        runner = None
    prog = NumbaTestProgram(module=None,
                            testRunner=runner, exit=False,
                            verbosity=verbosity,
                            nomultiproc=nomultiproc)
    return prog.result


def test(**kwargs):
    return run_tests(**kwargs).wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
