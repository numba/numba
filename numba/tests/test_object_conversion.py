import unittest

from numba.minivect import minitypes
from numba import *

@autojit(backend='ast')
def convert(obj_var, native_var):
    obj_var = native_var
    native_var = obj_var
    return native_var

class TestConversion(unittest.TestCase):
    def test_conversion(self):
        assert convert(object(), 10.2) == 10.2
        assert convert(object(), 10) == 10
        assert convert(object(), "foo") == "foo"
        obj = object()
        assert convert(object(), obj) == obj

if __name__ == "__main__":
    unittest.main()