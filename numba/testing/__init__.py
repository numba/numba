from __future__ import print_function, division, absolute_import

import os
import sys
import functools
import traceback
from fnmatch import fnmatch
from os.path import join, isfile, relpath, normpath, splitext

from .main import NumbaTestProgram, SerialSuite, make_tag_decorator
from numba import config
import numba.unittest_support as unittest


def load_testsuite(loader, dir):
    """Find tests in 'dir'."""
    try:
        suite = unittest.TestSuite()
        files = []
        for f in os.listdir(dir):
            path = join(dir, f)
            if isfile(path) and fnmatch(f, 'test_*.py'):
                files.append(f)
            elif isfile(join(path, '__init__.py')):
                suite.addTests(loader.discover(path))
        for f in files:
            # turn 'f' into a filename relative to the toplevel dir...
            f = relpath(join(dir, f), loader._top_level_dir)
            # ...and translate it to a module name.
            f = splitext(normpath(f.replace(os.path.sep, '.')))[0]
            suite.addTests(loader.loadTestsFromName(f))
        return suite
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)


def allow_interpreter_mode(fn):
    """Temporarily re-enable interpreter mode
    """
    @functools.wraps(fn)
    def _core(*args, **kws):
        config.COMPATIBILITY_MODE = True
        try:
            fn(*args, **kws)
        finally:
            config.COMPATIBILITY_MODE = False
    return _core


def run_tests(argv=None, defaultTest=None, topleveldir=None,
              xmloutput=None, verbosity=1, nomultiproc=False):
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
                            defaultTest=defaultTest,
                            topleveldir=topleveldir,
                            testRunner=runner, exit=False,
                            verbosity=verbosity,
                            nomultiproc=nomultiproc)
    return prog.result
