from __future__ import print_function, absolute_import, division

from ctypes import *
import sys

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


is_windows = sys.platform.startswith('win32')

if is_windows:
    libc = cdll.msvcrt
else:
    libc = CDLL(None)

# A typed libc function (cdecl under Windows)

c_sin = libc.sin
c_sin.argtypes = [c_double]
c_sin.restype = c_double

def use_c_sin(x):
    return c_sin(x)

# An untyped libc function

c_untyped = libc.cos

def use_c_untyped(x):
    return c_untyped(x)

# A libc function wrapped in a CFUNCTYPE

ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)

def use_ctype_wrapping(x):
    return ctype_wrapping(x)

# A Python API function

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

if is_windows:
    # A function with the stdcall calling convention
    c_sleep = windll.kernel32.Sleep
    c_sleep.argtypes = [c_uint]
    c_sleep.restype = None

    def use_c_sleep(x):
        c_sleep(x)


def use_c_pointer(x):
    """
    Running in Python will cause a segfault.
    """
    threadstate = savethread()
    x += 1
    restorethread(threadstate)
    return x


class TestCTypes(unittest.TestCase):

    def test_c_sin(self):
        pyfunc = use_c_sin
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    @unittest.skipUnless(is_windows, "Windows-specific test")
    def test_stdcall(self):
        # Just check that it doesn't crash
        cres = compile_isolated(use_c_sleep, [types.uintc])
        cfunc = cres.entry_point
        cfunc(1)

    def test_ctype_wrapping(self):
        pyfunc = use_ctype_wrapping
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    def test_ctype_voidptr(self):
        pyfunc = use_c_pointer
        # pyfunc will segfault if called
        cres = compile_isolated(pyfunc, [types.int32])
        cfunc = cres.entry_point
        x = 123
        self.assertTrue(cfunc(x), x + 1)

    def test_untyped_function(self):
        with self.assertRaises(TypeError) as raises:
            compile_isolated(use_c_untyped, [types.double])
        self.assertIn("ctypes function 'cos' doesn't define its argument types",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()

