"""
Test extending types via the numba.extending.* API.
"""

from numba import njit
from numba import types
from numba import cgutils
from numba.errors import TypingError
from numba.extending import lower_builtin
from numba.extending import models, register_model
from numba.extending import make_attribute_wrapper
from numba.extending import type_callable
from numba.extending import overload
from numba.extending import typeof_impl

from numba import unittest_support as unittest


class TestExtTypDummy(unittest.TestCase):

    def setUp(self):
        class Dummy(object):
            def __init__(self, value):
                self.value = value

        class DummyType(types.Type):
            def __init__(self):
                super(DummyType, self).__init__(name='Dummy')

        dummy_type = DummyType()

        @register_model(DummyType)
        class DummyModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                members = [
                    ('value', types.intp),
                    ]
                models.StructModel.__init__(self, dmm, fe_type, members)

        make_attribute_wrapper(DummyType, 'value', 'value')

        @type_callable(Dummy)
        def type_dummy(context):
            def typer(value):
                return dummy_type
            return typer

        @lower_builtin(Dummy, types.intp)
        def impl_dummy(context, builder, sig, args):
            typ = sig.return_type
            [value] = args
            dummy = cgutils.create_struct_proxy(typ)(context, builder)
            dummy.value = value
            return dummy._getvalue()

        @typeof_impl.register(Dummy)
        def typeof_dummy(val, c):
            return DummyType()

        # Store attributes
        self.Dummy = Dummy
        self.DummyType = DummyType

    def _add_float_overload(self):
        @overload(float)
        def dummy_to_float(x):
            if isinstance(x, self.DummyType):
                def codegen(x):
                    return float(x.value)
                return codegen
            else:
                raise TypeError('cannot type float({})'.format(x))

    def test_overload_float(self):
        self._add_float_overload()
        Dummy = self.Dummy

        @njit
        def foo(x):
            return float(Dummy(x))

        self.assertEqual(foo(123), float(123))

    def test_overload_float_error_msg(self):
        self._add_float_overload()

        @njit
        def foo(x):
            return float(x)

        with self.assertRaises(TypingError) as raises:
            foo(1j)

        self.assertIn("TypeError: float() does not support complex",
                      str(raises.exception))
        self.assertIn("TypeError: cannot type float(complex128)",
                      str(raises.exception))

    def test_unboxing(self):
        """A test for the unboxing logic on unknown type
        """
        Dummy = self.Dummy

        @njit
        def foo(x):
            # pass a dummy object into another function
            bar(Dummy(x))

        # make sure a cpython wrapper is created
        @njit(no_cpython_wrapper=False)
        def bar(dummy_obj):
            pass

        foo(123)
        with self.assertRaises(TypeError) as raises:
            bar(Dummy(123))
        self.assertIn("can't unbox Dummy type", str(raises.exception))

    def test_boxing(self):
        """A test for the boxing logic on unknown type
        """
        Dummy = self.Dummy

        @njit
        def foo(x):
            return Dummy(x)

        with self.assertRaises(TypeError) as raises:
            foo(123)
        self.assertIn("cannot convert native Dummy to Python object",
                      str(raises.exception))
