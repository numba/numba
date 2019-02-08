
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


def heappop(heap):
    return hq.heappop(heap)


def heappush(heap, item):
    return hq.heappush(heap, item)


def heapreplace(heap, item):
    return hq.heapreplace(heap, item)


def nsmallest(n, iterable):
    return hq.nsmallest(n, iterable)


def nlargest(n, iterable):
    return hq.nlargest(n, iterable)


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

        # includes non-finite elements
        element_pool = [3.142, -10.0, 5.5, np.nan, -np.inf, np.inf]

        # list which may contain duplicate elements
        for x in itertools.combinations_with_replacement(element_pool, 6):
            a = list(x)
            b = a[:]

            pyfunc(a)
            cfunc(b)
            self.assertPreciseEqual(a, b)

        # single element list
        for i in range(len(element_pool)):
            a = [element_pool[i]]
            b = a[:]

            pyfunc(a)
            cfunc(b)
            self.assertPreciseEqual(a, b)

        # elements are tuples
        a = [(3, 33), (1, 11), (2, 22)]
        b = a[:]
        pyfunc(a)
        cfunc(b)
        self.assertPreciseEqual(a, b)

    def check_invariant(self, heap):
        for pos, item in enumerate(heap):
            if pos:
                parentpos = (pos - 1) >> 1
                self.assertTrue(heap[parentpos] <= item)

    def test_heapify(self):
        # inspired by
        # https://github.com/python/cpython/blob/e42b7051/Lib/test/test_heapq.py
        pyfunc = heapify
        cfunc = jit(nopython=True)(pyfunc)

        for size in list(range(1, 30)) + [20000]:
            heap = self.rnd.random_sample(size).tolist()
            cfunc(heap)
            self.check_invariant(heap)

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

    def test_heappop_basic_sanity(self):
        pyfunc = heappop
        cfunc = jit(nopython=True)(pyfunc)

        def a_variations():
            yield [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
            yield [(3, 33), (1, 111), (2, 2222)]
            yield np.full(5, fill_value=np.nan).tolist()
            yield np.linspace(-10, -5, 100).tolist()

        for a in a_variations():
            heapify(a)
            b = a[:]

            for i in range(len(a)):
                val_py = pyfunc(a)
                val_c = cfunc(b)
                self.assertPreciseEqual(a, b)
                self.assertPreciseEqual(val_py, val_c)

    def iterables(self):
        yield [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
        a = np.linspace(-10, 2, 23)
        yield a.tolist()
        yield a[::-1].tolist()
        self.rnd.shuffle(a)
        yield a.tolist()

    def test_heappush_basic(self):
        pyfunc_push = heappush
        cfunc_push = jit(nopython=True)(pyfunc_push)

        pyfunc_pop = heappop
        cfunc_pop = jit(nopython=True)(pyfunc_pop)

        for iterable in self.iterables():
            expected = sorted(iterable)
            heap = [iterable.pop(0)]  # must initialise heap

            for value in iterable:
                cfunc_push(heap, value)

            got = [cfunc_pop(heap) for _ in range(len(heap))]
            self.assertPreciseEqual(expected, got)

    def test_nsmallest_basic(self):
        pyfunc = nsmallest
        cfunc = jit(nopython=True)(pyfunc)

        for iterable in self.iterables():
            for n in range(-5, len(iterable) + 3):
                expected = pyfunc(1, iterable)
                got = cfunc(1, iterable)
                self.assertPreciseEqual(expected, got)

    def test_nlargest_basic(self):
        pyfunc = nlargest
        cfunc = jit(nopython=True)(pyfunc)

        for iterable in self.iterables():
            for n in range(-5, len(iterable) + 3):
                expected = pyfunc(1, iterable)
                got = cfunc(1, iterable)
                self.assertPreciseEqual(expected, got)

    def test_heapreplace_basic(self):
        pyfunc = heapreplace
        cfunc = jit(nopython=True)(pyfunc)

        a = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

        heapify(a)
        b = a[:]

        for item in [-4, 4, 14]:
            pyfunc(a, item)
            cfunc(b, item)
            self.assertPreciseEqual(a, b)

        a = np.linspace(-3, 13, 20)
        a[4] = np.nan
        a[-1] = np.inf
        a = a.tolist()

        heapify(a)
        b = a[:]

        for item in [-4.0, 3.142, -np.inf, np.inf]:
            pyfunc(a, item)
            cfunc(b, item)
            self.assertPreciseEqual(a, b)
