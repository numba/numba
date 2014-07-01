from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types, typeinfer


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


if __name__ == '__main__':
    unittest.main()
