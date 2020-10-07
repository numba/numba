import unittest
from unittest import TestCase
from contextlib import contextmanager

import numpy as np

from numba import types
from numba.core.compiler import compile_isolated
from numba.tests.support import override_config


class TestRefOpPruning(TestCase):

    _numba_parallel_test_ = False

    def check(self, func, *argtys, **prune_types):
        """
        Asserts the the func compiled with argument types "argtys" reports
        refop pruning statistics that match those supplied as kwargs in
        **prune_types.
        """

        with override_config('EXPERIMENTAL_REFPRUNE_PASS', '1'):
            cres = compile_isolated(func, (*argtys,))

        pstats = cres.metadata.get('prune_stats', None)
        self.assertIsNotNone(pstats)

        for k, v in prune_types.items():
            stat = getattr(pstats, k, None)
            self.assertIsNotNone(stat)
            self.assertEqual(stat, v)

    @contextmanager
    def set_refprune_flags(self, flags):
        with override_config('EXPERIMENTAL_REFPRUNE_FLAGS', flags):
            yield

    def test_basic_block_1(self):
        # some nominally involved control flow and ops, there's only basic_block
        # opportunities present here.
        def func(n):
            a = np.zeros(n)
            acc = 0
            if n > 4:
                b = a[1:]
                acc += b[1]
            else:
                c = a[:-1]
                acc += c[0]
            return acc

        self.check(func, (types.intp), basicblock=16)

    def test_diamond_1(self):
        # most basic?! diamond
        def func(n):
            a = np.ones(n)
            x = 0
            if n > 2:
                x = a.sum()
            return x + 1

        # disable fanout pruning
        with self.set_refprune_flags('per_bb,diamond'):
            self.check(func, (types.intp), basicblock=41, diamond=2,
                       fanout=0, fanout_raise=0)

    def test_diamond_2(self):
        # more complex diamonds
        def func(n):
            con = []
            for i in range(n):
                con.append(np.arange(i))
            c = 0.0
            for arr in con:
                c += arr.sum() / (1 + arr.size)
            return c

        # disable fanout pruning
        with self.set_refprune_flags('per_bb,diamond'):
            self.check(func, (types.intp), basicblock=54, diamond=6,
                       fanout=0, fanout_raise=0)

    def test_fanout_1(self):
        # most basic?! fan-out
        def func(n):
            a = np.zeros(n)
            b = np.zeros(n)
            x = (a, b)
            acc = 0.
            for i in x:
                acc += i[0]
            return acc

        self.check(func, (types.intp), basicblock=44, fanout=3)


if __name__ == "__main__":
    unittest.main()
