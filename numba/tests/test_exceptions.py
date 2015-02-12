
import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import types
from numba import unittest_support as unittest
from numba.pythonapi import NativeError
from .support import TestCase


force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


class MyError(Exception):
    pass


def raise_class(exc):
    def raiser(i):
        if i == 1:
            raise exc
        return i
    return raiser


def raise_instance(exc, arg):
    def raiser(i):
        if i == 1:
            raise exc(arg)
        return i
    return raiser


def reraise():
    raise


class TestRaising(TestCase):

    def test_unituple_index_error(self):
        def pyfunc(a, i):
            return a.shape[i]

        cres = compile_isolated(pyfunc, (types.Array(types.int32, 1, 'A'),
                                         types.int32))

        cfunc = cres.entry_point
        a = np.empty(2)

        self.assertEqual(cfunc(a, 0), pyfunc(a, 0))

        with self.assertRaises(NativeError):
            cfunc(a, 2)

    def check_raise_class(self, flags):
        pyfunc = raise_class(MyError)
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        self.assertEqual(cfunc(0), 0)

        with self.assertRaises(MyError) as cm:
            cfunc(1)
        self.assertEqual(cm.exception.args, ())

    def test_raise_class_nopython(self):
        self.check_raise_class(flags=no_pyobj_flags)

    def test_raise_class_objmode(self):
        self.check_raise_class(flags=force_pyobj_flags)

    def check_raise_instance(self, flags):
        pyfunc = raise_instance(MyError, "some message")
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        self.assertEqual(cfunc(0), 0)

        with self.assertRaises(MyError) as cm:
            cfunc(1)
        self.assertEqual(cm.exception.args, ("some message",))

    def test_raise_instance_objmode(self):
        self.check_raise_instance(flags=force_pyobj_flags)

    def test_raise_instance_nopython(self):
        with self.assertTypingError():
            self.check_raise_instance(flags=no_pyobj_flags)

    def check_reraise(self, flags):
        pyfunc = reraise
        cres = compile_isolated(pyfunc, (), flags=flags)
        with self.assertRaises(ZeroDivisionError):
            try:
                1/0
            except ZeroDivisionError as e:
                pyfunc()

    def test_reraise_objmode(self):
        self.check_reraise(flags=force_pyobj_flags)

    def test_reraise_nopython(self):
        with self.assertTypingError():
            self.check_reraise(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
