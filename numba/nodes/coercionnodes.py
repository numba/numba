# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *
import numba.nodes

class CoercionNode(ExprNode):
    """
    Coerce a node to a different type
    """

    _fields = ['node']
    _attributes = ['type', 'name']

    def __new__(cls, node, dst_type, name=''):
        if isinstance(node, CoercionNode) and node.type == dst_type:
            return node

        return super(CoercionNode, cls).__new__(cls, node, dst_type, name=name)

    def __init__(self, node, dst_type, name=''):
        if node is self:
            # We are trying to coerce a CoercionNode which already has the
            # right type, so __new__ returns a CoercionNode, which then results
            # in __init__ being called
            return

        type = getattr(node, 'type', None) or node.variable.type
        if dst_type.is_pointer and type.is_int:
            assert type == Py_uintptr_t, type

        self.type = dst_type
        self.variable = Variable(dst_type)
        self.name = name

        self.node = self.verify_conversion(dst_type, node)

        if (dst_type.is_object and not node.variable.type.is_object and
                isinstance(node, numba.nodes.ArrayAttributeNode)):
            self.node = self.coerce_numpy_attribute(node)

    def coerce_numpy_attribute(self, node):
        """
        Numpy array attributes, such as 'data', get rewritten to direct
        accesses. Since they are being coerced back to objects, use a generic
        attribute access instead.
        """
        node = ast.Attribute(value=node.array, attr=node.attr_name,
                             ctx=ast.Load())
        node.variable = Variable(object_)
        node.type = object_
        return node

    @property
    def dst_type(self):
        """
        dst_type is always the same as type, and 'type' is kept consistent
        with Variable.type
        """
        return self.type

    @classmethod
    def coerce(cls, node_or_nodes, dst_type):
        if isinstance(node_or_nodes, list) and isinstance(dst_type, list):
            return [cls(node, dst) for node, dst in zip(node_or_nodes, dst_type)]
        elif isinstance(node_or_nodes, list):
            return [cls(node, dst_type) for node in node_or_nodes]
        return cls(node_or_nodes, dst_type)

    def verify_conversion(self, dst_type, node):
        if ((node.variable.type.is_complex or dst_type.is_complex) and
            (node.variable.type.is_object or dst_type.is_object)):
            if dst_type.is_complex:
                complex_type = dst_type
            else:
                complex_type = node.variable.type

            if not complex_type == complex128:
                node = CoercionNode(node, complex128)

        elif ((node.variable.type.is_datetime or dst_type.is_datetime) and
            (node.variable.type.is_object or dst_type.is_object)):
            if dst_type.is_datetime:
                datetime_type = dst_type
            else:
                datetime_type = node.variable.type

            if not datetime_type.is_datetime and \
                    not datetime_type.is_numpy_datetime:
                node = CoercionNode(node, datetime)

        elif ((node.variable.type.is_timedelta or dst_type.is_timedelta) and
            (node.variable.type.is_object or dst_type.is_object)):
            if dst_type.is_timedelta:
                timedelta_type = dst_type
            else:
                timedelta_type = node.variable.type

            if not timedelta_type.is_timedelta and \
                    not timedelta_type.is_timedelta:
                node = CoercionNode(node, timedelta)

        return node

    def __repr__(self):
        return "Coerce(%s, %s)" % (self.type, self.node)

class CastNode(ExprNode):
    """
    Explicit cast by user, e.g. double(value)
    """

    _fields = ["arg"]

    def __init__(self, node, type):
        self.arg = node
        self.type = type


class PromotionNode(ExprNode):
    """
    Coerces a variable of some type to another type for a phi node in a
    successor block.
    """

    _fields = ['node']

    def __init__(self, **kwargs):
        super(PromotionNode, self).__init__(**kwargs)
        self.variable = self.node.variable

class CoerceToObject(CoercionNode):
    "Coerce native values to objects"

class CoerceToNative(CoercionNode):
    "Coerce objects to native values"


class DeferredCoercionNode(ExprNode):
    """
    Coerce to the type of the given variable. The type of the variable may
    change in the meantime (e.g. may be promoted or demoted).
    """

    _fields = ['node']

    def __init__(self, node, variable):
        self.node = node
        self.variable = variable

class UntypedCoercion(ExprNode):
    """
    Coerce a node to the destination type. The node need not yet have a
    type or variable.
    """

    _fields = ['node']

    def __init__(self, node, type):
        self.node = node
        self.type = type
