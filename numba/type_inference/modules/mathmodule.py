"""
Resolve calls to math functions.

During type inference this produces MathNode nodes, and during
final specialization it produces LLVMIntrinsicNode and MathCallNode
nodes.
"""

import math
import cmath
import ctypes
import __builtin__ as builtins

from numba import *
from numba import nodes
from numba.minivect import minitypes
from numba import error
from numba import function_util
from numba.symtab import Variable
from numba import typesystem
from numba.typesystem import is_obj, promote_closest, get_type

from numba.type_inference.modules import utils

import llvm.core
import numpy as np

#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

register_math = utils.register_with_argchecking

def binop_type(context, x, y):
    "Binary result type for math operations"
    x_type = get_type(x)
    y_type = get_type(y)
    return context.promote_types(x_type, y_type)

def intrinsic_signature(type):
    if type.is_int:
        type = double

    signature = minitypes.FunctionType(return_type=type, args=[type, type])
    return signature

#----------------------------------------------------------------------------
# Categorize Calls as Math Calls
#----------------------------------------------------------------------------

def get_funcname(py_func):
    if py_func is np.abs:
        return 'fabs'
    elif py_func is np.round:
        return 'round'

    return py_func.__name__

def is_intrinsic(py_func):
    "Whether the math function is available as an llvm intrinsic"
    intrinsic_name = 'INTR_' + get_funcname(py_func).upper()
    is_intrinsic = hasattr(llvm.core, intrinsic_name)
    return is_intrinsic # and not is_win32

def math_suffix(name, type):
    if name == 'abs':
        name = 'fabs'

    if type.itemsize == 4:
        name += 'f' # sinf(float)
    elif type.itemsize == 16:
        name += 'l' # sinl(long double)
    return name

def is_math_function(func_args, py_func):
    if len(func_args) == 0 or len(func_args) > 1 or py_func is None:
        return False

    type = get_type(func_args[0])

    if type.is_array:
        type = type.dtype
        valid_type = type.is_float or type.is_int or type.is_complex
    else:
        valid_type = type.is_float or type.is_int

    math_name = get_funcname(py_func)
    is_math = math_name in libc_math_funcs
    if is_math and valid_type:
        math_name = math_suffix(math_name, type)
        is_math = filter_math_funcs([math_name])

    return valid_type and (is_intrinsic(py_func) or is_math)

def resolve_intrinsic(args, py_func, type):
    signature = intrinsic_signature(type)
    func_name = get_funcname(py_func).upper()
    return nodes.LLVMIntrinsicNode(signature, args, func_name=func_name)


#----------------------------------------------------------------------------
# Determine math functions
#----------------------------------------------------------------------------

is_win32 = sys.platform == 'win32'

def filter_math_funcs(math_func_names):
    if is_win32:
        dll = ctypes.cdll.msvcrt
    else:
        dll = ctypes.CDLL(None)

    result_func_names = []
    for name in math_func_names:
        if getattr(dll, name, None) is not None:
            result_func_names.append(name)

    return result_func_names

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
libc_math_funcs = filter_math_funcs(all_libc_math_funcs)

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
