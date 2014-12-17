from __future__ import print_function

import threading

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import jit
from .support import TestCase


def f(a):
    # This function has two characteristics:
    # - it runs long enough for threads to switch
    # - it can't be optimized too aggressively by LLVM, so loads and
    #   stores aren't eliminated (hopefully)
    for n in range(100000 // a.size):
        for idx in range(a.size):
            tmp = a[idx]
            a[a[idx] % a.size] = a[idx] + 1
            a[idx] = tmp

f_sig = "void(uint32[:])"


class TestGILRelease(TestCase):

    n_threads = 32

    def run_in_threads(self, func):
        # Run a workload in parallel, several times in a row, and collect
        # results.
        results = []
        for iteration in range(3):
            threads = []
            arr = np.arange(self.n_threads, dtype=np.uint32)
            for i in range(self.n_threads):
                t = threading.Thread(target=func, args=(arr,))
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            results.append(arr)
        return results

    def check_gil_held(self, func):
        arrays = self.run_in_threads(func)
        distinct = set(tuple(a) for a in arrays)
        self.assertEqual(len(distinct), 1, distinct)

    def check_gil_released(self, func):
        arrays = self.run_in_threads(func)
        distinct = set(tuple(a) for a in arrays)
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
        def caller(a):
            compiled_f(a)
        self.check_gil_released(caller)

    def test_gil_released_by_caller_and_callee(self):
        """
        Same, but with both caller and callee asking to release the GIL.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)
        @jit(f_sig, nopython=True, nogil=True)
        def caller(a):
            compiled_f(a)
        self.check_gil_released(caller)

    def test_gil_ignored_by_callee(self):
        """
        When only the callee asks to release the GIL, it gets ignored.
        """
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)
        @jit(f_sig, nopython=True)
        def caller(a):
            compiled_f(a)
        self.check_gil_held(caller)


if __name__ == '__main__':
    unittest.main()
