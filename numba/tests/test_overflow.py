"""
>>> native_convert(char, -10)
-10L
>>> native_convert(char, 10)
10L
>>> native_convert(char, 127)
127L

This doesn't work yet, we should get an error here. We don't get one because
autojit detects the int type which is natively truncated to a char.

TODO::::::::::

    >> native_convert(char, 128)

        => need exception!

>>> object_convert(char, 128)
Traceback (most recent call last):
    ...
OverflowError: value too large to convert to char
>>> object_convert(char, -128)
-128L
>>> object_convert(char, -129)
Traceback (most recent call last):
    ...
OverflowError: value too large to convert to char
>>> object_convert(char, 2.9)
2L

TODO:::::::::::

    Test all numeric types for overflows!

TODO:::::::::::

    Test typedef types (npy_intp, Py_uintptr_t, etc)
"""

from numba import *

@autojit
def native_convert(dst_type, value):
    return dst_type(value)

@autojit(locals=dict(obj=object_))
def object_convert(dst_type, obj):
    return dst_type(obj)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
