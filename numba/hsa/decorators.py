from __future__ import print_function, absolute_import, division
from numba import sigutils, types
from .compiler import compile_kernel, compile_device


def jit(signature, device=False):
    """JIT compile a python function conforming to
    the HSA-Python
    """
    if device:
        return _device_jit(signature)
    else:
        return _kernel_jit(signature)


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

