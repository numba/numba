# -*- coding: utf-8 -*-

"""
Extension method types.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem.types import NumbaType, make_polytype

#------------------------------------------------------------------------
# Extension Method Types
#------------------------------------------------------------------------

extra = ["name", "is_class_method", "is_static_method"]
_ExtMethodType = make_polytype(
    "extension_method", ["return_type", "args"] + extra,
    defaults=dict.fromkeys(extra))

class ExtMethodType(_ExtMethodType):

    @property
    def is_bound_method(self):
        return not (self.is_class_method or self.is_static_method)

class AutojitMethodType(NumbaType):
    is_autojit_method = True

#------------------------------------------------------------------------
# Method Signature Comparison
#------------------------------------------------------------------------

def drop_self(type):
    if type.is_static_method or type.is_class_method:
        return type.args

    assert len(type.args) >= 1 and type.args[0].is_extension
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
