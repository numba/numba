"""
Struct and complex nodes.

Structs are allocated on the stack, and not mutated as values. This is
because mutations are attribute or index assignments, which are not
recognized as variable assignments. Hence mutation cannot propagate new
values. So we mutate what we have on the stack.
"""

from numba.nodes import *

def struct_type(type):
    if type.is_reference:
        type = type.referenced_type

    return type

class StructAttribute(ExtTypeAttribute):
    # expr : = StructAttribute(expr, string, expr_context, Type, metadata)
    # metadata := StructAttribute | ComplexAttribute
    _fields = ['value']

    def __init__(self, value, attr, ctx, type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.ctx = ctx
        self.struct_type = type

        type = struct_type(type)
        self.attr_type = type.fielddict[attr]
        self.field_idx = type.fields.index((attr, self.attr_type))

        self.type = self.attr_type
        self.variable = Variable(self.type, promotable_type=False)


class StructVariable(Node):
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

class ComplexNode(Node):
    _fields = ['real', 'imag']

    type = complex128
    variable = Variable(type)

class ComplexAttributeNode(Node):

    _fields = ["value"]

    def __init__(self, value, attr):
        self.value = value
        self.attr = attr
        self.type = value.type.base_type
        self.variable = Variable(self.type)
