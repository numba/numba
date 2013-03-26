# -*- coding: utf-8 -*-

"""
Extension method types.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import *

#------------------------------------------------------------------------
# Extension Method Types
#------------------------------------------------------------------------

class ExtMethodType(NumbaType, minitypes.FunctionType):
    """
    Extension method type, a FunctionType plus the following fields:

        is_class_method: is classmethod?
        is_static_method: is staticmethod?
        is_bound_method: is bound method?
    """

    is_extension_method = True

    def __init__(self, return_type, args, name=None,
                 is_class=False, is_static=False, **kwds):
        super(ExtMethodType, self).__init__(return_type, args, name, **kwds)

        self.is_class_method = is_class
        self.is_static_method = is_static
        self.is_bound_method = not (is_class or is_static)

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
