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
import collections
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

n_ary_mathsyms = {
    'hypot'     : 2,
    'atan2'     : 2,
    'logaddexp' : 2,
    'logaddexp2': 2,
    'pow'       : (2, 3),
}

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

def mk_infer_math_call(default_result_type):
    def infer(typesystem, call_node, *args):
        "Resolve calls to llvmmath math calls"
        # signature is a generic signature, build a correct one
        type = reduce(typesystem.promote, map(get_type, call_node.args))

        if type.is_numeric and rank(type) < rank(default_result_type):
            type = default_result_type
        elif type.is_array and type.dtype.is_int:
            type = typesystem.array(double, type.ndim)

        call_node.args[:] = nodes.CoercionNode.coerce(call_node.args, type)

        # TODO: Remove the abuse below
        nodes.annotate(typesystem.env, call_node, is_math=True)
        call_node.variable = Variable(type)
        return call_node

    return infer

infer_math_call = mk_infer_math_call(double)
infer_cmath_call = mk_infer_math_call(complex128)

# ______________________________________________________________________
# abs()

def abs_(typesystem, node, x):
    from . import builtinmodule

    argtype = get_type(x)
    nodes.annotate(typesystem.env, node, is_math=True)

    if argtype.is_array and argtype.dtype.is_numeric:
        # Handle np.abs() on arrays
        dtype = builtinmodule.abstype(argtype.dtype)
        result_type = argtype.add('dtype', dtype)
        node.variable = Variable(result_type)
    else:
        node = builtinmodule.abs_(typesystem, node, x)

    return node

register_math_typefunc(1)(abs_, np.abs)

#----------------------------------------------------------------------------
# Register Type Functions
#----------------------------------------------------------------------------

def register_math(infer_math_call, nargs, value):
    register = register_math_typefunc(nargs)
    register(infer_math_call, value)

def npy_name(name):
    return math2ufunc.get(name, name)

id_name = lambda x: x

# ______________________________________________________________________

def reg(mod, register, getname):
    """Register all functions listed in mathsyms and n_ary_mathsyms"""
    nargs = lambda f: n_ary_mathsyms.get(f, 1)
    for symname in mathsyms + list(n_ary_mathsyms):
        if hasattr(mod, getname(symname)):
            register(nargs(symname), getattr(mod, getname(symname)))

reg(math, partial(register_math, infer_math_call), id_name)
reg(cmath, partial(register_math, infer_cmath_call), id_name)
reg(np, partial(register_math, infer_math_call), npy_name)