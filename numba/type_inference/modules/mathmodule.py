"""
Resolve calls to math functions.

During type inference this produces MathNode nodes, and during
final specialization it produces LLVMIntrinsicNode and MathCallNode
nodes.
"""

import math
import cmath
import __builtin__ as builtins

import numpy as np

from numba import *
from numba import nodes
from numba.symtab import Variable
from numba.typesystem import get_type
from numba.type_inference.modules import utils


#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

register_math = utils.register_with_argchecking

def binop_type(context, x, y):
    "Binary result type for math operations"
    x_type = get_type(x)
    y_type = get_type(y)
    return context.promote_types(x_type, y_type)

#----------------------------------------------------------------------------
# Determine math functions
#----------------------------------------------------------------------------

# sin(double), sinf(float), sinl(long double)
unary_libc_math_funcs = [
    'sin',
    'cos',
    'tan',
    'sqrt',
    'acos',
    'asin',
    'atan',
    'atan2',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'log',
    'log2',
    'log10',
    'fabs',
    'erfc',
    'ceil',
    'exp',
    'exp2',
    'expm1',
    'rint',
    'log1p',
]

n_ary_libc_math_funcs = [
    'pow',
    'round',
]

all_libc_math_funcs = unary_libc_math_funcs + n_ary_libc_math_funcs

#----------------------------------------------------------------------------
# Math Type Inferers
#----------------------------------------------------------------------------

# TODO: Move any rewriting parts to lowering phases

def infer_unary_math_call(context, call_node, arg):
    "Resolve calls to math functions to llvm.log.f32() etc"
    # signature is a generic signature, build a correct one
    type = get_type(call_node.args[0])

    if type.is_int:
        type = double
    elif type.is_array and type.dtype.is_int:
        type = type.copy(dtype=double)

    # signature = minitypes.FunctionType(return_type=type, args=[type])
    # result = nodes.MathNode(py_func, signature, call_node.args[0])
    nodes.annotate(context.env, call_node, is_math=True)
    call_node.variable = Variable(type)
    return call_node

# ______________________________________________________________________
# pow()

def pow_(context, call_node, node, power, mod=None):
    dst_type = binop_type(context, node, power)
    call_node.variable = Variable(dst_type)
    return call_node

register_math((2, 3), math.pow)
register_math(2, np.power)

# ______________________________________________________________________
# abs()

def abs_(context, node, x):
    import builtinmodule

    argtype = get_type(x)

    if argtype.is_array and argtype.is_numeric:
        # Handle np.abs() on arrays
        dtype = builtinmodule.abstype(argtype.dtype)
        result_type = argtype.copy(dtype=dtype)
        node.variable = Variable(result_type)
        return node

    return builtinmodule.abs_(context, node, x)

register_math(1, np.abs)

#----------------------------------------------------------------------------
# Register Type Functions
#----------------------------------------------------------------------------

def register(nargs, value):
    register = register_math(nargs)
    register(infer_unary_math_call, value)

def register_typefuncs():
    modules = [builtins, math, cmath, np]
    # print all_libc_math_funcs
    for libc_math_func in unary_libc_math_funcs:
        for module in modules:
            if hasattr(module, libc_math_func):
                register(1, getattr(module, libc_math_func))

register_typefuncs()
