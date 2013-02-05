"""
Resolve calls to math functions.

During type inference this produces MathNode nodes, and during
final specialization it produces LLVMIntrinsicNode and MathCallNode
nodes.
"""

import ctypes
import __builtin__ as builtins

from numba import *
from numba import nodes
from numba import error
from numba import function_util
from numba.symtab import Variable
from numba import typesystem
from numba.typesystem import is_obj, promote_closest

from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound)

import llvm.core
import numpy as np


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
libc_math_funcs = [
    'sin',
    'cos',
    'tan',
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
    'log2',
    'log10',
    'fabs',
    'pow',
    'erfc',
    'ceil',
    'expm1',
    'rint',
    'log1p',
    'round',
]

libc_math_funcs = filter_math_funcs(libc_math_funcs)

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

def is_math_function(func_args, py_func):
    if len(func_args) == 0 or len(func_args) > 1 or py_func is None:
        return False

    type = func_args[0].variable.type

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

def resolve_intrinsic(args, py_func, signature):
    func_name = get_funcname(py_func).upper()
    return nodes.LLVMIntrinsicNode(signature, args, func_name=func_name)

def math_suffix(name, type):
    if name == 'abs':
        name = 'fabs'

    if type.itemsize == 4:
        name += 'f' # sinf(float)
    elif type.itemsize == 16:
        name += 'l' # sinl(long double)
    return name

def _resolve_libc_math(self, args, py_func, signature):
    arg_type = signature.args[0]
    name = math_suffix(get_funcname(py_func), arg_type)
    return nodes.MathCallNode(signature, args, llvm_func=None,
                              py_func=py_func, name=name)

def resolve_math_call(call_node, py_func):
    "Resolve calls to math functions to llvm.log.f32() etc"
    # signature is a generic signature, build a correct one
    orig_type = type = call_node.args[0].variable.type

    if type.is_int:
        type = double
    elif type.is_array and type.dtype.is_int:
        type = type.copy(dtype=double)

    signature = minitypes.FunctionType(return_type=type, args=[type])
    result = nodes.MathNode(py_func, signature, call_node.args[0])
    return result

def binop_type(context, x, y):
    "Binary result type for math operations"
    x_type = x.variable.type
    y_type = y.variable.type
    dst_type = context.promote_types(x_type, y_type)
    type = dst_type
    if type.is_int:
        type = double

    signature = minitypes.FunctionType(return_type=type, args=[type, type])
    return dst_type, type, signature

def pow(context, node, power, mod=None):
    dst_type, pow_type, signature = binop_type(node, power)
    args = [node, power]
    if pow_type.is_float and mod is None:
        result = resolve_intrinsic(args, pow, signature)
    else:
        if mod is not None:
            args.append(mod)
        result = nodes.call_pyfunc(pow, args)

    return nodes.CoercionNode(result, dst_type)
