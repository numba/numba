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
            type = numba.typeof(pyval)

        # if pyval is not _NULL:
        #     assert not type.is_object

        self.variable = Variable(type, is_constant=True, constant_value=pyval)
        self.type = type
        self.pyval = pyval

    def __repr__(self):
        return "const(%s, %s)" % (self.pyval, self.type)

#------------------------------------------------------------------------
# NULL Constants
#------------------------------------------------------------------------

_NULL = object()
NULL_obj = ConstNode(_NULL, object_)
NULL = ConstNode(_NULL, void.pointer())
