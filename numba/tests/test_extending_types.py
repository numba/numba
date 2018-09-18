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
