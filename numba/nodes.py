import ast

from .symtab import Variable
from . import _numba_types as numba_types
from numba import utils

import llvm.core

context = utils.get_minivect_context()

class Node(ast.AST):
    """
    Superclass for Numba AST nodes
    """
    _fields = []

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

class CoercionNode(Node):
    _fields = ['node']
    def __init__(self, node, dst_type):
        self.node = node
        self.dst_type = dst_type
        self.variable = Variable(dst_type)

    @classmethod
    def coerce(cls, node_or_nodes, dst_type):
        if isinstance(node_or_nodes, list):
            return [cls(node, dst_type) for node in node_or_nodes]
        return cls(node_or_nodes, dst_type)

class DeferredCoercionNode(CoercionNode):
    """
    Coerce to the type of the given variable. The type of the variable may
    change in the meantime (e.g. may be promoted or demoted).
    """

    def __init__(self, node, variable):
        self.node = node
        self.variable = variable

class ConstNode(Node):
    def __init__(self, pyval, type=None):
        if type is None:
            type = context.typemapper.from_python(pyval)

        self.variable = Variable(type, is_constant=True, constant_value=pyval)
        self.type = type
        self.pyval = pyval

    def value(self, builder):
        type = self.type
        ltype = type.to_llvm(context)

        constant = self.pyval

        if type.is_float:
            lvalue = llvm.core.Constant.real(ltype, constant)
        elif type.is_int:
            lvalue = llvm.core.Constant.int(ltype, constant)
        elif type.is_complex:
            base_ltype = self.to_llvm(type.base_type)
            lvalue = llvm.core.Constant.struct([(base_ltype, constant.real),
                                                (base_ltype, constant.imag)])
        elif type.is_object:
            raise NotImplementedError
        elif type.is_function:
            # TODO:
            # lvalue = map_to_function(constant, type, self.mod)
            raise NotImplementedError
        else:
            raise NotImplementedError("Constant %s of type %s" %
                                                        (self.pyval, type))

        return lvalue
