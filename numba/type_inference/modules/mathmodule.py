# -*- coding: utf-8 -*-
"""
Resolve calls to math functions.

During type inference this annotates math calls, and during
final specialization it produces LLVMIntrinsicNode and MathCallNode
nodes.
"""
from __future__ import print_function, division, absolute_import

import math
import cmath
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import numpy as np

from numba import *
from numba import nodes
from numba.symtab import Variable
from numba.typesystem import get_type, rank
from numba.type_inference.modules import utils

#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

register_math_typefunc = utils.register_with_argchecking

def binop_type(typesystem, x, y):
    "Binary result type for math operations"
    x_type = get_type(x)
    y_type = get_type(y)
    return typesystem.promote(x_type, y_type)

#----------------------------------------------------------------------------
# Determine math functions
#----------------------------------------------------------------------------

# sin(double), sinf(float), sinl(long double)
mathsyms = [
    'sin',
    'cos',
    'tan',
    'sqrt',
    'acos',
    'asin',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'log',
    'log2',
    'log10',
    'erfc',
    'floor',
    'ceil',
    'exp',
    'exp2',
    'expm1',
    'rint',
    'log1p',
]

math2ufunc = {
    'asin' : 'arcsin',
    'acos' : 'arccos',
    'atan' : 'arctan',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'atan2': 'arctan2',
    'pow'  : 'power',
}

ufunc2math = dict((v, k) for k, v in math2ufunc.items())

#----------------------------------------------------------------------------
# Math Type Inferers
#----------------------------------------------------------------------------

# TODO: Move any rewriting parts to lowering phases

def infer_unary_math_call(typesystem, call_node, arg, default_result_type=double):
    "Resolve calls to llvmmath math calls"
    # signature is a generic signature, build a correct one
    type = get_type(call_node.args[0])

    print("infer...", arg)
    if type.is_numeric and rank(type) < rank(default_result_type):
        type = default_result_type
    elif type.is_array and type.dtype.is_int:
        type = typesystem.array(double, type.ndim)

    call_node.args[0] = nodes.CoercionNode(call_node.args[0], type)

    # TODO: Remove the abuse below
    nodes.annotate(typesystem.env, call_node, is_math=True)
    call_node.variable = Variable(type)
    return call_node

def infer_unary_cmath_call(typesystem, call_node, arg):
    result = infer_unary_math_call(typesystem, call_node, arg,
                                   default_result_type=complex128)
    nodes.annotate(typesystem.env, call_node, is_cmath=True)
    return result

# ______________________________________________________________________
# pow()

def pow_(typesystem, call_node, node, power, mod=None):
    dst_type = binop_type(typesystem, node, power)
    call_node.variable = Variable(dst_type)
    return call_node

register_math_typefunc((2, 3), math.pow)
register_math_typefunc(2, np.power)

# ______________________________________________________________________
# abs()

def abs_(typesystem, node, x):
    import builtinmodule

    argtype = get_type(x)

    if argtype.is_array and argtype.is_numeric:
        # Handle np.abs() on arrays
        dtype = builtinmodule.abstype(argtype.dtype)
        result_type = argtype.add('dtype', dtype)
        node.variable = Variable(result_type)
        return node

    return builtinmodule.abs_(typesystem, node, x)

register_math_typefunc(1, np.abs)

#----------------------------------------------------------------------------
# Register Type Functions
#----------------------------------------------------------------------------

def register_math(nargs, value):
    register = register_math_typefunc(nargs)
    register(infer_unary_math_call, value)

def register_cmath(nargs, value):
    register = register_math_typefunc(nargs)
    register(infer_unary_cmath_call, value)

def npy_name(name):
    return math2ufunc.get(name, name)

def register_typefuncs():
    modules = [builtins, math, cmath, np]
    register_funcs = [register_math, register_math,
                      register_cmath, register_math]
    id = lambda x: x
    namefuncs = [id, id, id, npy_name]

    for libc_math_func in mathsyms:
        for module, register, namefunc in zip(modules, register_funcs, namefuncs):
            if hasattr(module, namefunc(libc_math_func)):
                register(1, getattr(module, namefunc(libc_math_func)))

register_typefuncs()
