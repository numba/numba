from __future__ import print_function, absolute_import, division

from ctypes import *
import sys

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import jit, types


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

c_cos = libc.cos
c_cos.argtypes = [c_double]
c_cos.restype = c_double

def use_two_funcs(x):
    return c_sin(x) - c_cos(x)

# An untyped libc function

c_untyped = libc.exp

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


def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)


class TestCTypes(unittest.TestCase):

    def test_c_sin(self):
        pyfunc = use_c_sin
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

    def test_two_funcs(self):
        # Check that two constant functions don't get mixed up.
        pyfunc = use_two_funcs
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

    def test_function_pointer(self):
        pyfunc = use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [
            (c_sin, c_cos, 1.0),
            (c_sin, c_cos, -1.0),
            (c_cos, c_sin, 1.0),
            (c_cos, c_sin, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        # A single specialization was compiled for all calls
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_untyped_function(self):
        with self.assertRaises(TypeError) as raises:
            compile_isolated(use_c_untyped, [types.double])
        self.assertIn("ctypes function 'exp' doesn't define its argument types",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()

