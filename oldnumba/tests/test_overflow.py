"""
>>> native_convert(char, -10)
-10
>>> native_convert(char, 10)
10
>>> native_convert(char, 127)
127

This doesn't work yet, we should get an error here. We don't get one because
autojit detects the int type which is natively truncated to a char.

TODO::::::::::

    >> native_convert(char, 128)

        => need exception!

>>> object_convert(char, 128)
Traceback (most recent call last):
    ...
OverflowError: value too large to convert to signed char
>>> object_convert(char, -128)
-128
>>> object_convert(char, -129)
Traceback (most recent call last):
    ...
OverflowError: value too large to convert to signed char
>>> object_convert(char, 2.9)
2

TODO:::::::::::

    Test all numeric types for overflows!

TODO:::::::::::

    Test typedef types (npy_intp, Py_uintptr_t, etc)
"""

import unittest
from numba import *

@autojit
def native_convert(dst_type, value):
    return dst_type(value)

@autojit(locals=dict(obj=object_))
def object_convert(dst_type, obj):
    return dst_type(obj)

class TestConversion(unittest.TestCase):

    def test_native_conversion(self):
        assert native_convert(char, -10) == b'\xf6'
        assert native_convert(char, 10)  == b'\n'
        assert native_convert(char, 127) == b'\x7f'

        # TODO: the below should raise an exception
        # We don't get one because autojit detects the int type which is
        # simply truncated to a char
        # native_convert(char, 128)

    def test_object_conversion(self):
        assert object_convert(char, -128) == b'\x80'
        assert object_convert(char, 2.9)  == b'\x02'

    def test_overflow(self):
        self._convert_overflow(128 , char,   'signed char')
        self._convert_overflow(-129, char,   'signed char')
        self._convert_overflow(2**31, int32, 'signed int')

    def _convert_overflow(self, value, type, typename):
        self.assertRaises(OverflowError, object_convert, type, value)
        # self.assertEqual(captured.exception.args[0],
        #                  "value too large to convert to %s" % typename)

if __name__ == "__main__":
    unittest.main()
