"""
Test generic manipulation of objects.
"""


import unittest
from numba import njit
from numba.core.compiler import compile_isolated, Flags
from numba.core import types
from numba.tests.support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


class C(object):
    pass


def setattr_usecase(o, v):
    o.x = v

def delattr_usecase(o):
    del o.x


class TestAttributes(TestCase):
    def test_setattr(self, flags=enable_pyobj_flags):
        pyfunc = setattr_usecase
        cr = compile_isolated(pyfunc, (types.pyobject, types.int32), flags=flags)
        cfunc = cr.entry_point
        c = C()
        cfunc(c, 123)
        self.assertEqual(c.x, 123)

    def test_setattr_attribute_error(self, flags=enable_pyobj_flags):
        pyfunc = setattr_usecase
        cr = compile_isolated(pyfunc, (types.pyobject, types.int32), flags=flags)
        cfunc = cr.entry_point
        # Can't set undeclared slot
        with self.assertRaises(AttributeError):
            cfunc(object(), 123)

    def test_delattr(self, flags=enable_pyobj_flags):
        pyfunc = delattr_usecase
        cr = compile_isolated(pyfunc, (types.pyobject,), flags=flags)
        cfunc = cr.entry_point
        c = C()
        c.x = 123
        cfunc(c)
        with self.assertRaises(AttributeError):
            c.x

    def test_delattr_attribute_error(self, flags=enable_pyobj_flags):
        pyfunc = delattr_usecase
        cr = compile_isolated(pyfunc, (types.pyobject,), flags=flags)
        cfunc = cr.entry_point
        # Can't delete non-existing attribute
        with self.assertRaises(AttributeError):
            cfunc(C())


if __name__ == '__main__':
    unittest.main()
