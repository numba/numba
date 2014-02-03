from __future__ import print_function, absolute_import, division
from ctypes import *
import sys
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types

is_windows = sys.platform.startswith('win32')

if not is_windows:
    proc = CDLL(None)

    c_sin = proc.sin
    c_sin.argtypes = [c_double]
    c_sin.restype = c_double


    def use_c_sin(x):
        return c_sin(x)


    ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)


    def use_ctype_wrapping(x):
        return ctype_wrapping(x)



    savethread = pythonapi.PyEval_SaveThread
    savethread.argtypes = []
    savethread.restype = c_void_p

    restorethread = pythonapi.PyEval_RestoreThread
    restorethread.argtypes = [c_void_p]
    restorethread.restype = None


    def use_c_pointer(x):
        """
        Running in Python will cause a segfault.
        """
        threadstate = savethread()
        x += 1
        restorethread(threadstate)
        return x


@unittest.skipIf(is_windows, "Test not supported on windows")
class TestCTypes(unittest.TestCase):
    def test_c_sin(self):
        pyfunc = use_c_sin
        cres = compile_isolated(pyfunc, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

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

if __name__ == '__main__':
    unittest.main()

