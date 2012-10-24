import sys
import StringIO
import unittest
import functools

from nose.tools import nottest
import nose.plugins.skip
from numba.decorators import jit as jit_, autojit

import __builtin__

def checkSkipFlag(reason):
    def _checkSkipFlag(fn):
        @nottest
        def _checkSkipWrapper(self, *args, **kws):
            self.skipTest(reason)
        return _checkSkipWrapper
    return _checkSkipFlag

class ByteCodeTestCase(unittest.TestCase):
    jit = staticmethod(jit_)
    backend = 'bytecode'
    autojit = staticmethod(autojit(backend=backend))

def main():
    import sys, logging
    if '-d' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv.remove('-d')
    unittest.main()

class StdoutReplacer(object):
    def __enter__(self, *args):
        self.out = sys.stdout
        sys.stdout = StringIO.StringIO()
        return sys.stdout

    def __exit__(self, *args):
        sys.stdout = self.out