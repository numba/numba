# -*- coding: utf-8 -*-
"""
Type functions for Python builtins.
"""
from __future__ import print_function, division, absolute_import

from numba import *
from numba import nodes
from numba import error
# from numba import function_util
# from numba.specialize.mathcalls import is_math_function
from numba.symtab import Variable
from numba import typesystem
from numba.typesystem import get_type

from numba.type_inference.modules import utils

#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

register_builtin = utils.register_with_argchecking

def cast(node, dst_type):
    if len(node.args) == 0:
        return nodes.ConstNode(0, dst_type)
    else:
        return nodes.CoercionNode(node.args[0], dst_type=dst_type)

#----------------------------------------------------------------------------
# Type Functions for Builtins
#----------------------------------------------------------------------------

# TODO: add specializer functions to insert coercions before late specialization
# TODO: don't rewrite AST here

@register_builtin((1, 2, 3), can_handle_deferred_types=True)
def range_(context, node, start, stop, step):
    node.variable = Variable(typesystem.range_)
    node.args = nodes.CoercionNode.coerce(node.args, dst_type=Py_ssize_t)
    return node

if not PY3:
    @register_builtin((1, 2, 3), can_handle_deferred_types=True)
    def xrange_(context, node, start, stop, step):
        return range_(context, node, start, stop, step)

@register_builtin(1)
def len_(context, node, obj):
    # Simplify len(array) to ndarray.shape[0]
    argtype = get_type(obj)
    if argtype.is_array:
        shape_attr = nodes.ArrayAttributeNode('shape', node.args[0])
        new_node = nodes.index(shape_attr, 0)
        return new_node

    return Py_ssize_t

@register_builtin((0, 1, 2), can_handle_deferred_types=True)
def _int(context, node, x, base, dst_type=int_):
    # Resolve int(x) and float(x) to an equivalent cast

    if len(node.args) < 2:
        return cast(node, dst_type)

    node.variable = Variable(dst_type)
    return node

if not PY3:
    @register_builtin((0, 1, 2), can_handle_deferred_types=True)
    def _long(context, node, x, base):
        return _int(context, node, x, base)

@register_builtin((0, 1), can_handle_deferred_types=True)
def _float(context, node, x):
    return cast(node, double)

@register_builtin((0, 1, 2), can_handle_deferred_types=True)
def complex_(context, node, a, b):
    if len(node.args) == 2:
        args = nodes.CoercionNode.coerce(node.args, double)
        return nodes.ComplexNode(real=args[0], imag=args[1])
    else:
        return cast(node, complex128)

def abstype(argtype):
    if argtype.is_complex:
        result_type = double
    elif argtype.is_float or argtype.is_int:
        result_type = argtype
    else:
        result_type = object_

    return result_type

@register_builtin(1)
def abs_(context, node, x):
    node.variable = Variable(abstype(get_type(x)))
    return node

@register_builtin((2, 3))
def pow_(context, node, base, exponent, mod):
    from . import mathmodule
    return mathmodule.pow_(context, node, base, exponent)

@register_builtin((1, 2))
def round_(context, node, number, ndigits):
    # is_math = is_math_function(node.args, round)
    argtype = get_type(number)

    if len(node.args) == 1 and argtype.is_int:
        # round(myint) -> float(myint)
        return nodes.CoercionNode(node.args[0], double)

    if argtype.is_float or argtype.is_int:
        dst_type = double
    else:
        dst_type = object_
        node.args[0] = nodes.CoercionNode(node.args[0], object_)

    node.variable = Variable(dst_type)
    return node # nodes.CoercionNode(node, double)

@register_builtin(0)
def globals_(context, node):
    return typesystem.dict_
    # return nodes.ObjectInjectNode(func.__globals__)

@register_builtin(0)
def locals_(context, node):
    raise error.NumbaError("locals() is not supported in numba functions")
