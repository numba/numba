# -*- coding: utf-8 -*-
"""
Type functions for Python builtins.
"""
from __future__ import print_function, division, absolute_import

import ast
from numba import *
from numba import nodes
from numba import error
# from numba import function_util
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
def range_(typesystem, node, start, stop, step):
    node.variable = Variable(typesystem.range_)
    node.args = nodes.CoercionNode.coerce(node.args, dst_type=Py_ssize_t)
    return node

if not PY3:
    @register_builtin((1, 2, 3), can_handle_deferred_types=True)
    def xrange_(typesystem, node, start, stop, step):
        return range_(typesystem, node, start, stop, step)

@register_builtin(1)
def len_(typesystem, node, obj):
    # Simplify len(array) to ndarray.shape[0]
    argtype = get_type(obj)
    if argtype.is_array:
        shape_attr = nodes.ArrayAttributeNode('shape', node.args[0])
        new_node = nodes.index(shape_attr, 0)
        return new_node

    return Py_ssize_t

@register_builtin((0, 1, 2), can_handle_deferred_types=True)
def _int(typesystem, node, x, base, dst_type=int_):
    # Resolve int(x) and float(x) to an equivalent cast

    if len(node.args) < 2:
        return cast(node, dst_type)

    node.variable = Variable(dst_type)
    return node

if not PY3:
    @register_builtin((0, 1, 2), can_handle_deferred_types=True)
    def _long(typesystem, node, x, base):
        return _int(typesystem, node, x, base)

@register_builtin((0, 1), can_handle_deferred_types=True)
def _float(typesystem, node, x):
    return cast(node, double)

@register_builtin((0, 1), can_handle_deferred_types=True)
def _bool(context, node, x):
    return cast(node, bool_)

@register_builtin((0, 1, 2), can_handle_deferred_types=True)
def complex_(typesystem, node, a, b):
    if len(node.args) == 2:
        args = nodes.CoercionNode.coerce(node.args, double)
        return nodes.ComplexNode(real=args[0], imag=args[1])
    else:
        return cast(node, complex128)

def abstype(argtype):
    if argtype.is_complex:
        result_type = argtype.base_type
    elif argtype.is_float or argtype.is_int:
        result_type = argtype
    else:
        result_type = object_

    return result_type

@register_builtin(1)
def abs_(typesystem, node, x):
    node.variable = Variable(abstype(get_type(x)))
    return node

@register_builtin((2, 3))
def pow_(typesystem, node, base, exponent, mod):
    base_type = get_type(base)
    exp_type = get_type(exponent)
    node.variable = Variable(typesystem.promote(base_type, exp_type))
    return node
    # from . import mathmodule
    # return mathmodule.pow_(typesystem, node, base, exponent)

@register_builtin((1, 2))
def round_(typesystem, node, number, ndigits):
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

def minmax(typesystem, args, op):
    if len(args) < 2:
        return

    res = args[0]
    for arg in args[1:]:
        lhs_type = get_type(res)
        rhs_type = get_type(arg)
        res_type = typesystem.promote(lhs_type, rhs_type)
        if lhs_type != res_type:
            res = nodes.CoercionNode(res, res_type)
        if rhs_type != res_type:
            arg = nodes.CoercionNode(arg, res_type)

        lhs_temp = nodes.TempNode(res_type)
        rhs_temp = nodes.TempNode(res_type)
        res_temp = nodes.TempNode(res_type)
        lhs = lhs_temp.load(invariant=True)
        rhs = rhs_temp.load(invariant=True)
        expr = ast.IfExp(ast.Compare(lhs, [op], [rhs]), lhs, rhs)
        body = [
            ast.Assign([lhs_temp.store()], res),
            ast.Assign([rhs_temp.store()], arg),
            ast.Assign([res_temp.store()], expr),
        ]
        res = nodes.ExpressionNode(body, res_temp.load(invariant=True))

    return res

@register_builtin(None)
def min_(typesystem, node, *args):
    return minmax(typesystem, args, ast.Lt())

@register_builtin(None)
def max_(typesystem, node, *args):
    return minmax(typesystem, args, ast.Gt())

@register_builtin(0)
def globals_(typesystem, node):
    return typesystem.dict_of_obj
    # return nodes.ObjectInjectNode(func.__globals__)

@register_builtin(0)
def locals_(typesystem, node):
    raise error.NumbaError("locals() is not supported in numba functions")
