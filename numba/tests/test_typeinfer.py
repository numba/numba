from __future__ import print_function, division, absolute_import
import os, sys, subprocess
import itertools
import numpy as np
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types, typeinfer
from numba import typing


class TestArgRetCasting(unittest.TestCase):
    def test_arg_ret_casting(self):
        def foo(x):
            return x

        args = (types.int32,)
        return_type = types.float32
        cres = compile_isolated(foo, args, return_type)
        self.assertTrue(isinstance(cres.entry_point(123), float))
        self.assertEqual(cres.signature.args, args)
        self.assertEqual(cres.signature.return_type, return_type)

    def test_arg_ret_mismatch(self):
        def foo(x):
            return x

        args = (types.Array(types.int32, 1, 'C'),)
        return_type = types.float32
        try:
            cres = compile_isolated(foo, args, return_type)
        except typeinfer.TypingError as e:
            print("Exception raised:", e)
        else:
            self.fail("Should complain about array casting to float32")


class TestTupleUnify(unittest.TestCase):
    def test_int_tuple_unify(self):
        """
        Test issue #493
        """

        def foo(an_int32, an_int64):
            a = an_int32, an_int32
            while True:  # infinite loop
                a = an_int32, an_int64
            return a

        args = (types.int32, types.int64)
        # Check if compilation is successful
        cres = compile_isolated(foo, args)


class TestUnify(unittest.TestCase):
    @staticmethod
    def _actually_test_complex_unify():
        def pyfunc(a):
            res = 0.0
            for i in range(len(a)):
                res += a[i]
            return res

        argtys = [types.Array(types.complex128, 1, 'C')]
        cres = compile_isolated(pyfunc, argtys)
        return (pyfunc, cres)

    def test_complex_unify_issue599(self):
        pyfunc, cres = self._actually_test_complex_unify()
        arg = np.array([1.0j])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(arg), pyfunc(arg))

    def test_complex_unify_issue599_multihash(self):
        """
        Test issue #599 for multiple values of PYTHONHASHSEED.
        """
        env = os.environ.copy()
        for seedval in (1, 2, 1024):
            env['PYTHONHASHSEED'] = str(seedval)
            subproc = subprocess.Popen(
                [sys.executable, '-c',
                 'import numba.tests.test_typeinfer as test_mod\n' +
                 'test_mod.TestUnify._actually_test_complex_unify()'],
                env=env)
            subproc.wait()
            self.assertEqual(subproc.returncode, 0, 'Child process failed.')

    def unify_pair_test(self, n):
        """
        Test all permutations of N-combinations of numeric types and ensure
        that the unification matches
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, n):
            res = [ctx.unify_types(*comb)
                   for comb in itertools.permutations(tys)]
            # All result must be equal
            first_result = res[0]
            for other in res[1:]:
                self.assertEqual(first_result, other)

    def test_unify_pair(self):
        self.unify_pair_test(2)
        self.unify_pair_test(3)

    def test_bitwidth_number_types(self):
        """All numeric types have bitwidth attribute
        """
        for ty in types.number_domain:
            self.assertTrue(hasattr(ty, "bitwidth"))

if __name__ == '__main__':
    unittest.main()
