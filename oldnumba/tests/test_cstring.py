#! /usr/bin/env python
# ______________________________________________________________________

from numba import *
from numba import string_ as cstring, int_

from nose.tools import nottest
from numba.testing import test_support

# ______________________________________________________________________

def convert(input_str):
    return int(input_str[0:5])

# ______________________________________________________________________

def fast_convert(input_str):
    with nopython:
        return int(input_str[0:5])

# ______________________________________________________________________

class TestCString(test_support.ASTTestCase):
    def test_convert(self, **kws):
        jit_convert = self.jit(argtypes = (cstring,), restype = int_, **kws)(
            convert)
        for exp in xrange(10):
            test_str = str(10 ** exp)
            self.assertEqual(jit_convert(test_str), convert(test_str))

    def test_convert_nopython(self, **kws):
        jit_convert = self.jit(argtypes = (cstring,), restype = int_, **kws)(
            fast_convert)
        for exp in xrange(10):
            test_str = str(10 ** exp)
            self.assertEqual(jit_convert(test_str), convert(test_str))

# ______________________________________________________________________

if __name__ == "__main__":
#    TestCString('test_convert').debug()
    test_support.main()

# ______________________________________________________________________
# End of test_cstring.py
