from __future__ import print_function

import ctypes
import ctypes.util
import os
import sys
import threading

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import jit
from .support import TestCase


# This CPython API function is a portable way to get the current thread id.
PyThread_get_thread_ident = ctypes.pythonapi.PyThread_get_thread_ident
PyThread_get_thread_ident.restype = ctypes.c_long
PyThread_get_thread_ident.argtypes = []

# A way of sleeping from nopython code
if os.name == 'nt':
    sleep = ctypes.windll.kernel32.Sleep
    sleep.argtypes = [ctypes.c_uint]
    sleep.restype = None
    sleep_factor = 1  # milliseconds
else:
    sleep = ctypes.CDLL(ctypes.util.find_library("c")).usleep
    sleep.argtypes = [ctypes.c_uint]
    sleep.restype = ctypes.c_int
    sleep_factor = 1000  # milliseconds


def f(a, offset):
    # If run from one thread at a time, the function will always fill the
    # array with identical values.
    # If run from several threads at a time, the function will probably
    # fill the array with differing values.
    for idx in range(a.size):
        # Let another thread run
        sleep(1 * sleep_factor)
        a[(idx + offset) % a.size] = PyThread_get_thread_ident()

f_sig = "void(int64[:], intp)"


class TestGILRelease(TestCase):

    n_threads = 2

    def run_in_threads(self, func):
        # Run the function in parallel over an array and collect results.
        threads = []
        arr = np.arange(30, dtype=np.int64)
        for i in range(self.n_threads):
            # Ensure different threads have equally distributed start offsets
            # into the array.
            offset = i * arr.size // self.n_threads
            t = threading.Thread(target=func, args=(arr, offset))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return arr

    def check_gil_held(self, func):
        arr = self.run_in_threads(func)
        distinct = set(arr)
        self.assertEqual(len(distinct), 1, distinct)

    def check_gil_released(self, func):
        arr = self.run_in_threads(func)
        distinct = set(arr)
        self.assertGreater(len(distinct), 1, distinct)

    def test_gil_held(self):
        """
        Test the GIL is held by default, by checking serialized runs
        produce deterministic results.
        """
        cfunc = jit(f_sig, nopython=True)(f)
        self.check_gil_held(cfunc)

    def test_gil_released(self):
        """
        Test releasing the GIL, by checking parallel runs produce
        unpredictable results.
        """
        cfunc = jit(f_sig, nopython=True, nogil=True)(f)
        self.check_gil_released(cfunc)

    def test_gil_released_by_caller(self):
        """
        Releasing the GIL in the caller is sufficient to have it
        released in a callee.
        """
        compiled_f = jit(f_sig, nopython=True)(f)
        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_released_by_caller_and_callee(self):
        """
        Same, but with both caller and callee asking to release the GIL.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)
        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_ignored_by_callee(self):
        """
        When only the callee asks to release the GIL, it gets ignored.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)
        @jit(f_sig, nopython=True)
        def caller(a, i):
            compiled_f(a, i)
        self.check_gil_held(caller)


if __name__ == '__main__':
    unittest.main()
