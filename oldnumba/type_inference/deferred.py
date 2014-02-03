# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import translate, utils, typesystem
from numba.symtab import Variable

def create_deferred(type_inferer, node, deferred_cls):
    "Create a deferred type for an AST node"
    variable = Variable(None)
    deferred_type = deferred_cls(variable, type_inferer, node)
    variable.type = deferred_type
    node.variable = variable
    return deferred_type

def create_deferred_call(type_inferer, arg_types, call_node):
    "Set the ast.Call as uninferable for now"
    deferred_type = create_deferred(type_inferer, call_node,
                                    typesystem.DeferredCallType)
    for arg, arg_type in zip(call_node.args, arg_types):
        if arg_type.is_unresolved:
            deferred_type.dependences.append(arg)

    deferred_type.update()
    return call_node
