
from __future__ import print_function, absolute_import, division

import heapq as hq
import itertools

import numpy as np

from numba import jit
from numba.compiler import Flags
from .support import TestCase, CompilationCache, MemoryLeakMixin

no_pyobj_flags = Flags()
no_pyobj_flags.set("nrt")


def heapify(x):
    return hq.heapify(x)


class TestHeapq(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(TestHeapq, self).setUp()
        self.ccache = CompilationCache()
        self.rnd = np.random.RandomState(42)

    def test_heapify_basic_sanity(self):
        pyfunc = heapify
        cfunc = jit(nopython=True)(pyfunc)

        a = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
        b = a[:]

        pyfunc(a)
        cfunc(b)
        self.assertPreciseEqual(a, b)

    def test_heapify(self):
        pyfunc = heapify
        cfunc = jit(nopython=True)(pyfunc)

        element_pool = [3.142, -10.0, 5.5, np.nan, -np.inf, np.inf]

        for x in itertools.combinations_with_replacement(element_pool, 6):
            a = list(x)
            b = a[:]

            pyfunc(a)
            cfunc(b)
            self.assertPreciseEqual(a, b)

    def test_heapify_exceptions(self):
        pyfunc = heapify
        cfunc = jit(nopython=True)(pyfunc)

        # Exceptions leak references
        self.disable_leak_check()

        with self.assertTypingError() as e:
            cfunc((1, 5, 4))

        msg = 'heap argument must be a list'
        self.assertIn(msg, str(e.exception))

        with self.assertTypingError() as e:
            cfunc([1 + 1j, 2 - 3j])

        msg = ("'<' not supported between instances "
               "of 'complex' and 'complex'")
        self.assertIn(msg, str(e.exception))
