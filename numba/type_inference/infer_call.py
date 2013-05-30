# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numba
from numba import *
from numba import error, nodes
from numba.type_inference import module_type_inference
from numba import typesystem

if PY3:
    import builtins
else:
    import __builtin__ as builtins

debug = False
#debug = True

def resolve_function(func_variable):
    "Get a function object given a function name"
    func = None
    func_type = func_variable.type

    if func_type.is_builtin:
        func = getattr(builtins, func_variable.name)
    elif func_type.is_global:
        func = func_type.value
    elif func_type.is_module_attribute:
        func = getattr(func_type.module, func_type.attr)
    elif func_type.is_autojit_function:
        func = func_type.autojit_func
    elif func_type.is_jit_function:
        func = func_type.jit_func

    return func

def infer_typefunc(context, call_node, func_type, default_node):
    func_var = call_node.func.variable
    if func_var.is_constant:
        func_type = typesystem.known_value(func_var.constant_value)

    if (func_type.is_known_value and
            module_type_inference.is_registered(func_type.value)):
        # Try the module type inferers
        result_node = module_type_inference.resolve_call_or_none(
                                    context, call_node, func_type)
        if result_node:
            return result_node

    return default_node

def parse_signature(node, func_type):
    types = []
    for arg in node.args:
        if not arg.variable.type.is_cast:
            raise error.NumbaError(arg, "Expected a numba type")
        else:
            types.append(arg.variable.type)

    signature = func_type.dst_type(*types)
    new_node = nodes.const(signature, typesystem.meta(signature))
    return new_node
