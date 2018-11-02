
import numpy as np
import sys
import traceback

from numba.compiler import compile_isolated, Flags
from numba import jit, types
from numba import unittest_support as unittest
from .support import TestCase, tag

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()

class MyError(Exception):
    pass

class OtherError(Exception):
    pass

def raise_class(exc):
    def raiser(i):
        if i == 1:
            raise exc
        elif i == 2:
            raise ValueError
        elif i == 3:
            # The exception type is looked up on a module (issue #1624)
            raise np.linalg.LinAlgError
        return i
    return raiser

def raise_instance(exc, arg):
    def raiser(i):
        if i == 1:
            raise exc(arg, 1)
        elif i == 2:
            raise ValueError(arg, 2)
        elif i == 3:
            raise np.linalg.LinAlgError(arg, 3)
        return i
    return raiser

def reraise():
    raise

def outer_function(inner):
    def outer(i):
        if i == 3:
            raise OtherError("bar", 3)
        return inner(i)
    return outer

def assert_usecase(i):
    assert i == 1, "bar"


class TestRaising(TestCase):

    def test_unituple_index_error(self):
        def pyfunc(a, i):
            return a.shape[i]

        cres = compile_isolated(pyfunc, (types.Array(types.int32, 1, 'A'),
                                         types.int32))

        cfunc = cres.entry_point
        a = np.empty(2, dtype=np.int32)

        self.assertEqual(cfunc(a, 0), pyfunc(a, 0))

        with self.assertRaises(IndexError) as cm:
            cfunc(a, 2)
        self.assertEqual(str(cm.exception), "tuple index out of range")

    def check_against_python(self, exec_mode, pyfunc, cfunc,
                             expected_error_class, *args):

        assert exec_mode in (force_pyobj_flags, no_pyobj_flags)

        # invariant of mode, check the error class and args are the same
        with self.assertRaises(expected_error_class) as pyerr:
            pyfunc(*args)
        with self.assertRaises(expected_error_class) as jiterr:
            cfunc(*args)
        self.assertEqual(pyerr.exception.args, jiterr.exception.args)

        # in npm check bottom of traceback matches as frame injection with
        # location info should ensure this
        if exec_mode is no_pyobj_flags:

            # we only care about the bottom two frames, the error and the 
            # location it was raised.
            try:
                pyfunc(*args)
            except BaseException as e:
                py_frames = traceback.format_exception(*sys.exc_info())
                expected_frames = py_frames[-2:]

            try:
                cfunc(*args)
            except BaseException as e:
                c_frames = traceback.format_exception(*sys.exc_info())
                got_frames = c_frames[-2:]

            # check exception and the injected frame are the same
            for expf, gotf in zip(expected_frames, got_frames):
                self.assertEqual(expf, gotf)


    def check_raise_class(self, flags):
        pyfunc = raise_class(MyError)
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        self.assertEqual(cfunc(0), 0)
        self.check_against_python(flags, pyfunc, cfunc, MyError, 1)
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
        self.check_against_python(flags, pyfunc, cfunc,
                                  np.linalg.linalg.LinAlgError, 3)

    @tag('important')
    def test_raise_class_nopython(self):
        self.check_raise_class(flags=no_pyobj_flags)

    def test_raise_class_objmode(self):
        self.check_raise_class(flags=force_pyobj_flags)

    def check_raise_instance(self, flags):
        pyfunc = raise_instance(MyError, "some message")
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point

        self.assertEqual(cfunc(0), 0)
        self.check_against_python(flags, pyfunc, cfunc, MyError, 1)
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
        self.check_against_python(flags, pyfunc, cfunc,
                                  np.linalg.linalg.LinAlgError, 3)

    def test_raise_instance_objmode(self):
        self.check_raise_instance(flags=force_pyobj_flags)

    @tag('important')
    def test_raise_instance_nopython(self):
        self.check_raise_instance(flags=no_pyobj_flags)

    def check_raise_nested(self, flags, **jit_args):
        """
        Check exception propagation from nested functions.
        """
        inner_pyfunc = raise_instance(MyError, "some message")
        pyfunc = outer_function(inner_pyfunc)
        inner_cfunc = jit(**jit_args)(inner_pyfunc)
        cfunc = jit(**jit_args)(outer_function(inner_cfunc))

        self.check_against_python(flags, pyfunc, cfunc, MyError, 1)
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
        self.check_against_python(flags, pyfunc, cfunc, OtherError, 3)

    def test_raise_nested_objmode(self):
        self.check_raise_nested(force_pyobj_flags, forceobj=True)

    @tag('important')
    def test_raise_nested_nopython(self):
        self.check_raise_nested(no_pyobj_flags, nopython=True)

    def check_reraise(self, flags):
        pyfunc = reraise
        cres = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cres.entry_point
        def gen_impl(fn):
            def impl():
                try:
                    1/0
                except ZeroDivisionError as e:
                    fn()
            return impl
        pybased = gen_impl(pyfunc)
        cbased = gen_impl(cfunc)
        self.check_against_python(flags, pybased, cbased, ZeroDivisionError,)

    def test_reraise_objmode(self):
        self.check_reraise(flags=force_pyobj_flags)

    @tag('important')
    def test_reraise_nopython(self):
        self.check_reraise(flags=no_pyobj_flags)

    def check_raise_invalid_class(self, cls, flags):
        pyfunc = raise_class(cls)
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        with self.assertRaises(TypeError) as cm:
            cfunc(1)
        self.assertEqual(str(cm.exception),
                         "exceptions must derive from BaseException")

    def test_raise_invalid_class_objmode(self):
        self.check_raise_invalid_class(int, flags=force_pyobj_flags)
        self.check_raise_invalid_class(1, flags=force_pyobj_flags)

    def test_raise_invalid_class_nopython(self):
        with self.assertTypingError():
            self.check_raise_invalid_class(int, flags=no_pyobj_flags)
        with self.assertTypingError():
            self.check_raise_invalid_class(1, flags=no_pyobj_flags)

    def check_assert_statement(self, flags):
        pyfunc = assert_usecase
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        cfunc(1)
        self.check_against_python(flags, pyfunc, cfunc, AssertionError, 2)

    def test_assert_statement_objmode(self):
        self.check_assert_statement(flags=force_pyobj_flags)

    def test_assert_statement_nopython(self):
        self.check_assert_statement(flags=no_pyobj_flags)

    def check_raise_from_exec_string(self, flags):
        # issue #3428
        f_text = "def f(a):\n  assert a != 1"
        loc = {}
        exec(f_text, {}, loc)
        pyfunc = loc['f']
        cres = compile_isolated(pyfunc, (types.int32,), flags=flags)
        cfunc = cres.entry_point
        cfunc(2)
        self.check_against_python(flags, pyfunc, cfunc, AssertionError, 1)

    def test_assert_from_exec_string_objmode(self):
        self.check_raise_from_exec_string(flags=force_pyobj_flags)

    def test_assert_from_exec_string_nopython(self):
        self.check_raise_from_exec_string(flags=no_pyobj_flags)



if __name__ == '__main__':
    unittest.main()
