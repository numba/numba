"""
Tests the rewrite pass numba.rewrites.truthy_branches which wraps all predicates
in a call to `operator.truth`.
"""
from __future__ import print_function

import itertools
import operator

import numpy as np

from numba.compiler import run_frontend
from numba import errors, njit
from numba import rewrites, ir, jit
from .support import TestCase, MemoryLeakMixin


def compile_to_ir(func, rewrite=False):
    func_ir = run_frontend(func)

    class MockPipeline(object):
        def __init__(self, func_ir):
            self.typingctx = None
            self.targetctx = None
            self.args = None
            self.func_ir = func_ir
            self.typemap = None
            self.return_type = None
            self.calltypes = None

    if rewrite:
        # call this to get print etc rewrites
        rewrites.rewrite_registry.apply('before-inference', MockPipeline(func_ir),
                                        func_ir)
    return func_ir


class TestBranchPredicateRewrite(MemoryLeakMixin, TestCase):
    # Checks that the branch predicate gets wrapped in a call to operator.truth

    def check_branches(self, func_ir, mutation_expected=False):
        for _, blk in func_ir.blocks.items():
            branches = blk.find_insts(ir.Branch)
            for br in branches:
                defn = func_ir.get_definition(br.cond)
                if mutation_expected:
                    self.assertEqual(defn.op, 'call')
                    func = func_ir.get_definition(defn.func)
                    self.assertEqual(func.value, operator.truth)
                else:
                    self.assertFalse('truth_res' in str(br.cond))

    def check(self, func, *args, **kwargs):
        # check IR rewrite
        initial_ir = compile_to_ir(func)
        self.check_branches(initial_ir)
        rewritten_ir = compile_to_ir(func, rewrite=True)
        self.check_branches(rewritten_ir, True)

        objm = kwargs.get('check_obj_fallback', False)

        if not objm:
            # check it npm jits ok
            expected = func(*args)
            got = njit(func)(*args)
            self.assertEqual(expected, got)
        else:
            # check obj mode fallback path
            expected = func(*args)
            jfn = jit(func)
            got = jfn(*args)
            self.assertEqual(expected, got)
            self.assertEqual(jfn.nopython_signatures, [])

    def test_simple(self):

        def foo(x):
            if x:
                pass

        self.check(foo, 1)
        self.check(foo, 0)
        self.check(foo, None)
        self.check(foo, (1,))
        self.check(foo, ())

    def test_multiple_branches(self):

        # identical function apart from return type to trigger npm/objmode
        def foo_obj(x, y):
            acc = 0
            if x:
                acc += 1
            if y:
                acc += 2
            if x == 3:
                acc += 3
            if x or y:
                acc += 5
            if x or y:
                acc += 7
            if x is None:
                acc += 100
            else:
                if y is not None:
                    if x + 1 > 3:
                        acc += 1000
                    elif x / y > 12:
                        acc += 2000
                    else:
                        acc += 3000
                else:
                    acc += 10000
            return []

        def foo_npm(x, y):
            acc = 0
            if x:
                acc += 1
            if y:
                acc += 2
            if x == 3:
                acc += 3
            if x or y:
                acc += 5
            if x or y:
                acc += 7
            if x is None:
                acc += 100
            else:
                if y is not None:
                    if x + 1 > 3:
                        acc += 1000
                    elif x / y > 12:
                        acc += 2000
                    else:
                        acc += 3000
                else:
                    acc += 10000
            return 1

        for fn in [foo_npm, foo_obj]:
            fb = 'obj' in str(fn)
            self.check(fn, np.float64(0.0), 3, check_obj_fallback=fb)
            self.check(fn, 0, 3, check_obj_fallback=fb)
            self.check(fn, 1, 3, check_obj_fallback=fb)
            self.check(fn, None, None, check_obj_fallback=fb)
            self.check(fn, 3, 0, check_obj_fallback=fb)
            self.check(fn, None, 1, check_obj_fallback=fb)
            self.check(fn, 1, None, check_obj_fallback=fb)
            self.check(fn, None, 0, check_obj_fallback=fb)
            self.check(fn, 0, None, check_obj_fallback=fb)

    def test_multiple_branches_mixed_type(self):

        def foo(abool, atuple, alist, afloat, anint, optional_none=None):
            acc = 0
            if abool:
                acc += 1
            if atuple:
                acc += 2
            if alist:
                acc += 3
            if afloat:
                acc += 5
            if anint:
                acc += 7
            if optional_none:
                acc += 11
            return acc

        bools = [True, False]
        tuples = [(1,), ()]
        lists = [[1, ], ]
        floats = [np.float64(0), np.float64(1)]
        ints = [np.int64(0), np.int64(1)]
        noneorthing = [None, 1j]

        for args in itertools.product(bools, tuples, lists, floats, ints,
                                      noneorthing):
            self.check(foo, *args)

    def test_array_predicate_fail(self):

        @njit
        def foo(x):
            if x:
                return 1
            else:
                return 0

        with self.assertRaises(errors.TypingError) as e:
            foo(np.zeros(10))

        msg = 'truth value of an array with more than one element is ambiguous'
        self.assertIn(msg, str(e.exception))

    def test_valid_predicate_not_double_wrapped(self):
        # the use of operator.truth and truth as a function should both be
        # spotted and not rewrapped
        from operator import truth

        def foo(x):
            if operator.truth(x):
                return 1
            else:
                return 0

        def bar(x):
            if truth(x):
                return 1
            else:
                return 0

        for fn in foo, bar:
            initial_ir = compile_to_ir(fn)
            self.check_branches(initial_ir)
            # force rewrite pass
            rewritten_ir = compile_to_ir(fn, rewrite=True)
            # check no mutation
            self.check_branches(rewritten_ir, mutation_expected=False)
