from __future__ import print_function, absolute_import, division
from numba import sigutils, types
from .compiler import compile_kernel


def jit(restype=None, argtypes=None, device=False, **kws):
    restype, argtypes = _convert_types(restype, argtypes)

    if restype and not device and restype != types.void:
        raise TypeError("CUDA kernel must have void return type.")

    def kernel_jit(func):
        kernel = compile_kernel(func, argtypes)
        return kernel

    def device_jit(func):
        return compile_device(func, restype, argtypes)

    if device:
        return device_jit
    else:
        return kernel_jit


def _convert_types(restype, argtypes):
    # eval type string
    if sigutils.is_signature(restype):
        assert argtypes is None
        argtypes, restype = sigutils.normalize_signature(restype)

    return restype, argtypes

