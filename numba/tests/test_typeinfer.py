from __future__ import print_function, division, absolute_import
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
    def test_complex_unify_issue599(self):
        def pyfunc(a):
            res = 0.0
            for i in range(len(a)):
                res += a[i]
            return res

        arg = np.array([1.0j])
        argtys = [types.Array(types.complex128, 1, 'C')]
        cres = compile_isolated(pyfunc, argtys)
        cfunc = cres.entry_point
        self.assertEqual(cfunc(arg), pyfunc(arg))

    def test_unify_pair(self):
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, 2):
            res = [ctx.unify_types(*comb)
                   for comb in itertools.permutations(tys)]
            self.assertTrue(all(res[0] == other for other in res[1:]))

        for tys in itertools.combinations(types.number_domain, 3):
            print(tys)
            res = []
            for comb in itertools.permutations(tys):
                unified = ctx.unify_types(*comb)
                print(comb, '->', unified)
                res.append(unified)
            print(res)
            self.assertTrue(all(res[0] == other for other in res[1:]))

if __name__ == '__main__':
    unittest.main()
