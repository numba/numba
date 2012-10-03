#! /usr/bin/env python
# ______________________________________________________________________

from numba import *
from numba import c_string_type as cstring, int_

from numba.tests import test_support

# ______________________________________________________________________

def convert(input_str):
    return int(input_str[0:5])

# ______________________________________________________________________

class CStringTests(object):
    def test_convert(self):
        jit_convert = self.jit(argtypes = (cstring,), restype = int_)(
            convert)
        for exp in xrange(10):
            test_str = str(10 ** exp)
            self.assertEqual(jit_convert(test_str), convert(test_str))

# ______________________________________________________________________    

class TestBytecodeCString(test_support.ByteCodeTestCase, CStringTests):
    @test_support.checkSkipFlag("C strings not supported in bytecode "
                                "translator.")
    def test_convert(self, *args, **kws):
        return super(TestBytecodeCString, self).test_convert(*args, **kws)

# ______________________________________________________________________

class TestASTCString(test_support.ASTTestCase, CStringTests):
    @test_support.checkSkipFlag("Not implemented yet.")
    def test_convert(self, *args, **kws):
        return super(TestASTCString, self).test_convert(*args, **kws)

# ______________________________________________________________________

if __name__ == "__main__":
    test_support.main()

# ______________________________________________________________________
# End of test_cstring.py
