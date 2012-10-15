import unittest

from numba.minivect import minitypes
from numba import *

@autojit(backend='ast')
def convert(obj_var, native_var):
    obj_var = native_var
    native_var = obj_var
    return native_var

@autojit(locals=dict(obj=object_))
def convert_float(obj):
    var = float_(obj)
    return object_(var)

@autojit(locals=dict(obj=object_))
def convert_numeric(obj, dst_type):
    var = dst_type(obj)
    return object_(var)

class TestConversion(unittest.TestCase):
    def test_conversion(self):
        assert convert(object(), 10.2) == 10.2
        assert convert(object(), 10) == 10
        assert convert(object(), "foo") == "foo"
        obj = object()
        assert convert(object(), obj) == obj
        assert convert(object(), 10.2 + 5j) == 10.2 + 5j

        assert convert_float(10.5) == 10.5

    def test_numeric_conversion(self):
        types = [
            int_,
            long_,
            longlong,
            ulong,
            Py_ssize_t,
            # size_t,
            float_,
            double,
#            longdouble,
            complex64,
            complex128
        ]
        value = 2.5
        for dst_type in types:
            print dst_type
            if dst_type.is_int:
                expected = 2
            else:
                expected = 2.5

            result = convert_numeric(value, dst_type)
            assert result == expected, (result, dst_type)

if __name__ == "__main__":
#    print convert_float(10.2)
#    print convert(object(), 10.0)
    print convert_numeric(2.5, longdouble)
#    unittest.main()