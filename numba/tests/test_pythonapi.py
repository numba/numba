import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit


@intrinsic
def _pyapi_bytes_as_string(typingctx, csrc, size):
    sig = types.voidptr(csrc, size)  # cstring == void*

    def codegen(context, builder, sig, args):
        [csrc, size] = args
        api = context.get_python_api(builder)
        b = api.bytes_from_string_and_size(csrc, size)
        return api.bytes_as_string(b)
    return sig, codegen


def PyBytes_AsString(uni):
    # test_PyBytes_AsString will call this function with a unicode type.
    # We then use the underlying buffer to create a PyBytes object and call the
    # PyBytes_AsString function with PyBytes object as argument
    return _pyapi_bytes_as_string(uni._data, uni._length)


class TestPythonAPI(unittest.TestCase):

    def test_PyBytes_AsString(self):
        cfunc = jit(nopython=True)(PyBytes_AsString)
        cstr = cfunc('hello')  # returns a cstring

        fn = ctypes.pythonapi.PyBytes_FromString
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.py_object
        obj = fn(cstr)

        # Use the cstring created from bytes_as_string to create a python
        # bytes object
        self.assertEqual(obj, b'hello')


if __name__ == '__main__':
    unittest.main()
