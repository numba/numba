from __future__ import print_function, absolute_import, division
from numba import sigutils, types
from .compiler import (compile_kernel, compile_device, AutoJitHSAKernel,
                       compile_device_template)


def jit(signature=None, device=False):
    """JIT compile a python function conforming to
    the HSA-Python
    """
    if signature is None:
        return autojit(device=device)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(device=device)(func)
    else:
        if device:
            return _device_jit(signature)
        else:
            return _kernel_jit(signature)


def autojit(device=False):
    if device:
        return _device_autojit
    else:
        return _kernel_autojit


def _device_jit(signature):
    argtypes, restype = sigutils.normalize_signature(signature)

    def _wrapped(pyfunc):
        return compile_device(pyfunc, restype, argtypes)

    return _wrapped


def _kernel_jit(signature):
    argtypes, restype = sigutils.normalize_signature(signature)
    if restype is not None and restype != types.void:
        msg = "HSA kernel must have void return type but got {restype}"
        raise TypeError(msg.format(restype=restype))

    def _wrapped(pyfunc):
        return compile_kernel(pyfunc, argtypes)

    return _wrapped


def _device_autojit(pyfunc):
    return compile_device_template(pyfunc)


def _kernel_autojit(pyfunc):
    return AutoJitHSAKernel(pyfunc)



