import unittest
import functools

from nose.tools import nottest
from numba.decorators import jit as jit_, autojit

import __builtin__

def checkSkipFlag(reason):
    def _checkSkipFlag(fn):
        @nottest
        def _checkSkipWrapper(self, *args, **kws):
            if hasattr(__builtin__, '__noskip__'):
                return fn(self, *args, **kws)
            else:
                self.skipTest(reason)
        return _checkSkipWrapper
    return _checkSkipFlag

class ByteCodeTestCase(unittest.TestCase):
    jit = staticmethod(jit_)

class ASTTestCase(ByteCodeTestCase):
    jit = staticmethod(lambda *args, **kw: jit_(*args, **dict(kw, backend='ast')))

def main():
    import sys, logging
    if '-d' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv.remove('-d')
    unittest.main()
