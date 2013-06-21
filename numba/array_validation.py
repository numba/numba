# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast

from numba import error
from numba.ndarray_helpers import NumpyArray

is_object = lambda t: t.is_object and not t.is_array


class ArrayValidator(ast.NodeVisitor):
    """
    Validate array usage, depending on array representation
    (i.e. numpy vs. LLArray)
    """

    def __init__(self, env):
        self.env = env
        self.foreign = not issubclass(env.crnt.array, NumpyArray)

    def visit_CoercionNode(self, node):
        t1, t2 = node.type, node.node.type
        if self.foreign and t1.is_array ^ t2.is_array:
            raise error.NumbaError(node, "Cannot coerce non-numpy array %s" %
                                         self.env.crnt.array)

    visit_CoerceToObject = visit_CoercionNode
    visit_CoerceToNative = visit_CoercionNode

    def visit_MultiArrayAPINode(self, node):
        if self.foreign:
            signature = node.signature
            types = (signature.return_type,) + signature.argtypes
            for ty in types:
                if not ty.is_array:
                    raise TypeError("Cannot pass array %s as NumPy array" %
                                                        self.env.crnt.array)

    # TODO: Calling other numba functions with different array
    # TODO: representations may corrupt things