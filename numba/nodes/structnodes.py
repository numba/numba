# -*- coding: utf-8 -*-
"""
Struct and complex nodes.

Structs are allocated on the stack, and not mutated as values. This is
because mutations are attribute or index assignments, which are not
recognized as variable assignments. Hence mutation cannot propagate new
values. So we mutate what we have on the stack.
"""
from __future__ import print_function, division, absolute_import

from numba.nodes import *

def struct_type(type):
    if type.is_reference:
        type = type.referenced_type

    return type

class StructAttribute(ExprNode):
    # expr : = StructAttribute(expr, string, expr_context, Type, metadata)
    # metadata := StructAttribute | ComplexAttribute
    _fields = ['value']

    def __init__(self, value, attr, ctx, type, **kwargs):
        super(StructAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.ctx = ctx
        self.struct_type = type

        type = struct_type(type)
        self.attr_type = type.fielddict[attr]

        self.type = self.attr_type
        self.variable = Variable(self.type, promotable_type=False)

    @property
    def field_idx(self):
        fields = struct_type(self.struct_type).fields
        return fields.index((self.attr, self.attr_type))

class StructVariable(ExprNode):
    """
    Tells the type inferencer that the node is actually a valid struct that
    we can mutate. For instance

        func().a = 2

    is wrong if func() returns a struct by value. So we only allow references
    like struct.a = 2 and array[i].a = 2.
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(StructVariable, self).__init__(**kwargs)
        self.node = node
        self.type = node.type

class ComplexNode(ExprNode):
    _fields = ['real', 'imag']

    type = complex128
    variable = Variable(type)

    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

class ComplexAttributeNode(ExprNode):

    _fields = ["value"]

    def __init__(self, value, attr):
        self.value = value
        self.attr = attr
        self.type = value.type.base_type
        self.variable = Variable(self.type)

class DateTimeNode(ExprNode):

    _fields = ['timestamp', 'units']

    type = datetime()
    variable = Variable(type)

    def __init__(self, timestamp, units):
        self.timestamp = timestamp
        self.units = units

class DateTimeAttributeNode(ExprNode):
    
    _fields = ['value']

    def __init__(self, value, attr):
        self.value = value
        self.attr = attr
        self.type = value.type
        self.variable = Variable(self.type)

class NumpyDateTimeNode(ExprNode):

    _fields = ['datetime_string']

    type = datetime()
    variable = Variable(type)

    def __init__(self, datetime_string):
        self.datetime_string = datetime_string

class TimeDeltaNode(ExprNode):

    _fields = ['diff', 'units']

    type = timedelta()
    variable = Variable(type)

    def __init__(self, diff, units):
        self.diff = diff
        self.units = units

class NumpyTimeDeltaNode(ExprNode):

    _fields = ['diff', 'units_str']

    type = timedelta()
    variable = Variable(type)

    def __init__(self, diff, units_str):
        self.diff = diff
        self.units_str = units_str

class TimeDeltaAttributeNode(ExprNode):
    
    _fields = ['value']

    def __init__(self, value, attr):
        self.value = value
        self.attr = attr
        self.type = value.type
        self.variable = Variable(self.type)

