# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
import ctypes

import llvm.core
import numpy as np

from numba import nodes, double
from numba.type_inference.modules import mathmodule
from numba.typesystem import get_type

is_win32 = sys.platform == 'win32'

#----------------------------------------------------------------------------
# Categorize Calls as Math Calls
#----------------------------------------------------------------------------

def intrinsic_signature(nargs, type):
    if type.is_int or type.is_complex:
        type = double

    return type(*[type] * nargs)

def get_funcname(py_func):
    if py_func in (abs, np.abs):
        return 'fabs'
    elif py_func is np.round:
        return 'round'

    return py_func.__name__


def is_intrinsic(py_func):
    "Whether the math function is available as an llvm intrinsic"
    intrinsic_name = 'INTR_' + get_funcname(py_func).upper()
    is_intrinsic = hasattr(llvm.core, intrinsic_name)
    return is_intrinsic and not is_win32




def have_impl(math_name):
    return filter_math_funcs([math_name])

def have_double_impl(math_name):
    """
    Check whether we have an implementation of the math function in libc
    for type double (e.g. logf or logl may not be available, but log may be).
    """
    return math_suffix(math_name, double) in libc_math_funcs

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
        actual_math_name = math_suffix(math_name, type)
        is_math = have_impl(actual_math_name) or have_double_impl(math_name)

    return valid_type and (is_intrinsic(py_func) or is_math)


def resolve_intrinsic(args, py_func, type):
    signature = intrinsic_signature(len(args), type)
    func_name = get_funcname(py_func).upper()
    return nodes.LLVMIntrinsicNode(signature, args, func_name=func_name)


def resolve_libc_math(args, py_func, type):
    signature = intrinsic_signature(len(args), type)
    math_name = get_funcname(py_func)
    name = math_suffix(math_name, type)

    use_double_impl = not have_impl(name)
    if use_double_impl:
        assert have_double_impl(math_name)
        signature = double(*[double] * len(args))
        name = math_suffix(math_name, double)

    result = nodes.MathCallNode(signature, args, llvm_func=None,
                                py_func=py_func, name=name)
    return nodes.CoercionNode(result, type)

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

libc_math_funcs = filter_math_funcs(mathmodule.all_libc_math_funcs)
#print libc_math_funcs
#print filter_math_funcs(['log', 'logf', 'logl'])
