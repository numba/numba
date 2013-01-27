import sys
import StringIO
import unittest
import functools

from nose.tools import nottest
import nose.plugins.skip
import numba
from numba import *
import doctest_support

jit_ = jit

import __builtin__

class ASTTestCase(unittest.TestCase):
    jit = staticmethod(lambda *args, **kw: jit_(*args, **dict(kw, backend='ast')))
    backend = 'ast'
    autojit = staticmethod(autojit(backend=backend))

#------------------------------------------------------------------------
# Support for unittest in < py2.7
#------------------------------------------------------------------------

have_unit_skip = sys.version_info[:2] > (2, 6)

if have_unit_skip:
    from unittest import SkipTest
else:
    class SkipTest(Exception):
        "Skip a test in < py27"

@nottest
def skip_test(reason):
    if have_unit_skip:
        raise SkipTest(reason)
    else:
        print >>sys.stderr, "Skipping: " + reason

def skip_if(should_skip, message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if should_skip:
                skip_test(message)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def skip_unless(should_skip, message):
    return skip_if(not should_skip, message)

def skip(message):
    return skip_if(True, message)

def checkSkipFlag(reason):
    def _checkSkipFlag(fn):
        @nottest
        def _checkSkipWrapper(self, *args, **kws):
            skip_test(reason)
        return _checkSkipWrapper
    return _checkSkipFlag

#------------------------------------------------------------------------
# Test running
#------------------------------------------------------------------------

def main():
    import sys, logging
    if '-d' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv.remove('-d')
    if '-D' in sys.argv:
        logging.getLogger().setLevel(logging.NOTSET)
        sys.argv.remove('-D')
    unittest.main()

class StdoutReplacer(object):
    def __enter__(self, *args):
        self.out = sys.stdout
        sys.stdout = StringIO.StringIO()
        return sys.stdout

    def __exit__(self, *args):
        sys.stdout = self.out

from bytecode.test_support import ByteCodeTestCase

def testmod(module=None, runit=False):
    """
    Tests a doctest modules with numba functions. When run in nosetests, only
    populates module.__test__, when run as main, runs the doctests.
    """
    if module is None:
        modname = sys._getframe(1).f_globals['__name__']
        module = __import__(modname)
    else:
        modname = module.__name__

    doctest_support.testmod(module, run_doctests=runit or modname == '__main__')
    #if modname == '__main__':
    #numba.nose_run(mod)