from __future__ import print_function, absolute_import, division
from numba import sigutils, types
from .compiler import (compile_kernel, JitDPPyKernel, compile_dppy_func_template,
                       compile_dppy_func, get_ordered_arg_access_types)


def kernel(signature=None, access_types=None, debug=False):
    """JIT compile a python function conforming using the DPPy backend.

    A kernel is equvalent to an OpenCL kernel function, and has the
    same restrictions as definined by SPIR_KERNEL calling convention.
    """
    if signature is None:
        return autojit(debug=False, access_types=access_types)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(debug=False, access_types=access_types)(func)
    else:
        return _kernel_jit(signature, debug, access_types)


def autojit(debug=False, access_types=None):
    def _kernel_autojit(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        return JitDPPyKernel(pyfunc, ordered_arg_access_types)
    return _kernel_autojit


def _kernel_jit(signature, debug, access_types):
    argtypes, restype = sigutils.normalize_signature(signature)
    if restype is not None and restype != types.void:
        msg = ("DPPy kernel must have void return type but got {restype}")
        raise TypeError(msg.format(restype=restype))

    def _wrapped(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        return compile_kernel(None, pyfunc, argtypes, ordered_arg_access_types, debug)

    return _wrapped



def func(signature=None):
    if signature is None:
        return _func_autojit
    elif not sigutils.is_signature(signature):
        func = signature
        return _func_autojit(func)
    else:
        return _func_jit(signature)


def _func_jit(signature):
    argtypes, restype = sigutils.normalize_signature(signature)

    def _wrapped(pyfunc):
        return compile_dppy_func(pyfunc, restype, argtypes)

    return _wrapped

def _func_autojit(pyfunc):
    return compile_dppy_func_template(pyfunc)
