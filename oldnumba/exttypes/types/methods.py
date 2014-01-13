# -*- coding: utf-8 -*-

"""
Extension method types.
"""

from __future__ import print_function, division, absolute_import
from numba.typesystem import types, numbatypes

#------------------------------------------------------------------------
# Extension Method Types
#------------------------------------------------------------------------

class ExtMethodType(types.function):
    typename = "extmethod"
    argnames = ["return_type", "args", ("name", None), ("is_vararg", False),
                ("is_class_method", False), ("is_static_method", False)]
    flags = ["function", "object"]

    @property
    def is_bound_method(self):
        return not (self.is_class_method or self.is_static_method)

class AutojitMethodType(types.NumbaType):
    typename = "autojit_extmethod"
    flags = ["object"]

#------------------------------------------------------------------------
# Method Signature Comparison
#------------------------------------------------------------------------

def drop_self(type):
    if type.is_static_method or type.is_class_method:
        return type.args

    assert len(type.args) >= 1 and type.args[0].is_extension, type
    return type.args[1:]

def equal_signature_args(t1, t2):
    """
    Compare method signatures without regarding the 'self' type (which is
    set to the base extension type in the base class, and the derived
    extension type in the derived class).
    """
    return (t1.is_static_method == t2.is_static_method and
            t1.is_class_method == t2.is_class_method and
            t1.is_bound_method == t2.is_bound_method and
            drop_self(t1) == drop_self(t2))

def equal_signatures(t1, t2):
    return (equal_signature_args(t1, t2) and
            t1.return_type == t2.return_type)

def extmethod_to_function(ty):
    return numbatypes.function(ty.return_type, ty.args, ty.name)