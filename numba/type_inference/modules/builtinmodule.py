"""
Type functions for Python builtins.
"""

import functools
import __builtin__ as builtins


from numba import *
from numba import nodes
from numba import error
from numba import function_util
from numba.symtab import Variable
from numba import typesystem
from numba.typesystem import is_obj, promote_closest, get_type

from numba.type_inference.modules import mathmodule
from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound,
                                                        register_value)

#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

def _expect_n_args(node, name, nargs):
    if not isinstance(nargs, tuple):
        nargs = (nargs,)

    if len(node.args) not in nargs:
        expected = " or ".join(map(str, nargs))
        raise error.NumbaError(
            node, "builtin %s expects %s arguments" % (name,
                                                       expected))

def register_builtin(nargs):
    if not isinstance(nargs, tuple):
        nargs = (nargs,)

    def decorator(func):
        @functools.wraps(func)
        def infer(context, node, *args):
            _expect_n_args(node, name, nargs)

            need_nones = max(nargs) - len(args)
            args += (None,) * need_nones

            return func(context, node, *args)

        name = infer.__name__.strip("_")
        value = getattr(builtins, name)

        register_value(value, infer, pass_in_types=False, pass_in_callnode=True)

        return func # wrapper

    return decorator

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

@register_builtin((1, 2, 3))
def range_(context, node, start, stop, step):
    node.variable = Variable(typesystem.RangeType())
    node.args = nodes.CoercionNode.coerce(node.args, dst_type=Py_ssize_t)
    return node

if not PY3:
    @register_builtin((1, 2, 3))
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

@register_builtin((0, 1, 2))
def _int(context, node, x, base, dst_type=int_):
    # Resolve int(x) and float(x) to an equivalent cast

    if len(node.args) < 2:
        return cast(node, dst_type)

    node.variable = Variable(dst_type)
    return node

if not PY3:
    @register_builtin((0, 1, 2))
    def _long(context, node, x, base):
        return _int(context, node, x, base)

@register_builtin((0, 1))
def _float(context, node, x):
    return cast(node, double)

@register_builtin((0, 1, 2))
def complex_(context, node, a, b):
    if len(node.args) == 2:
        args = nodes.CoercionNode.coerce(node.args, double)
        return nodes.ComplexNode(real=args[0], imag=args[1])
    else:
        return cast(node, complex128)

@register_builtin(1)
def abs_(context, node, x):
    # Result type of the substitution during late
    # specialization
    result_type = object_
    argtype = get_type(x)

    # What we actually get back regardless of implementation,
    # e.g. abs(complex) goes throught the object layer, but we know the result
    # will be a double
    dst_type = argtype

    is_math = mathmodule.is_math_function(node.args, abs)

    if argtype.is_complex:
        dst_type = double
    elif is_math and (argtype.is_float or argtype.is_int):
        result_type = argtype

    node.variable = Variable(result_type)
    return nodes.CoercionNode(node, dst_type)

@register_builtin((2, 3))
def pow_(context, node, base, exponent, mod):
    return mathmodule.pow_(context, *node.args)

@register_builtin((1, 2))
def round_(context, node, number, ndigits):
    is_math = mathmodule.is_math_function(node.args, round)
    argtype = get_type(number)

    if len(node.args) == 1 and argtype.is_int:
        # round(myint) -> float(myint)
        return nodes.CoercionNode(node.args[0], double)

    if (argtype.is_float or argtype.is_int) and is_math:
        dst_type = argtype
    else:
        dst_type = object_
        node.args[0] = nodes.CoercionNode(node.args[0], object_)

    node.variable = Variable(dst_type)
    return nodes.CoercionNode(node, double)

@register_builtin(0)
def globals_(context, node):
    return typesystem.dict_
    # return nodes.ObjectInjectNode(func.func_globals)

@register_builtin(0)
def locals_(context, node):
    raise error.NumbaError("locals() is not supported in numba functions")
