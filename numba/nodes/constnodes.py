# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *
import numba.nodes

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def get_pointer_address(value, type):
    if type.is_known_pointer:
        return type.address
    else:
        return value

def is_null_constant(constant):
    return constant is _NULL

#------------------------------------------------------------------------
# Constant Nodes
#------------------------------------------------------------------------

class ConstNode(ExprNode):
    """
    Wrap a constant.
    """

    _attributes = ['type', 'pyval']

    def __init__(self, pyval, type=None):
        if type is None:
            type = context.typemapper.from_python(pyval)

        # if pyval is not _NULL:
        #     assert not type.is_object

        self.variable = Variable(type, is_constant=True, constant_value=pyval)
        self.type = type
        self.pyval = pyval

    def cast(self, dst_type):
        # This should probably happen in a transform !
        if dst_type.is_int:
            caster = int
        elif dst_type.is_float:
            caster = float
        elif dst_type.is_complex:
            caster = complex
        else:
            raise NotImplementedError(dst_type)

        try:
            self.pyval = caster(self.pyval)
        except ValueError:
            if dst_type.is_int and self.type.is_c_string:
                raise
            raise minierror.UnpromotableTypeError((dst_type, self.type))

        self.type = dst_type
        self.variable.type = dst_type

    def __repr__(self):
        return "const(%s, %s)" % (self.pyval, self.type)

#------------------------------------------------------------------------
# NULL Constants
#------------------------------------------------------------------------

_NULL = object()
NULL_obj = ConstNode(_NULL, object_)
NULL = ConstNode(_NULL, void.pointer())
