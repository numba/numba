from __future__ import print_function, absolute_import, division
from numba import sigutils, types
from .compiler import (compile_kernel, compile_device, AutoJitOCLKernel,
                       compile_device_template)


def jit(signature=None, device=False, debug=False):
    """JIT compile a python function conforming to
    the OCL-Python
    """
    if signature is None:
        return autojit(device=device, debug=False)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(device=device, debug=False)(func)
    else:
        if device:
            return _device_jit(signature, debug)
        else:
            return _kernel_jit(signature, debug)


def autojit(device=False, debug=False):
    if device:
        return _device_autojit
    else:
        return _kernel_autojit


def _device_jit(signature, debug):
    argtypes, restype = sigutils.normalize_signature(signature)

    def _wrapped(pyfunc):
        return compile_device(pyfunc, restype, argtypes, debug)

    return _wrapped


def _kernel_jit(signature, debug):
    argtypes, restype = sigutils.normalize_signature(signature)
    if restype is not None and restype != types.void:
        msg = "OCL kernel must have void return type but got {restype}"
        raise TypeError(msg.format(restype=restype))

    def _wrapped(pyfunc):
        return compile_kernel(pyfunc, argtypes, debug)

    return _wrapped


def _device_autojit(pyfunc):
    return compile_device_template(pyfunc)


def _kernel_autojit(pyfunc):
    return AutoJitOCLKernel(pyfunc)
