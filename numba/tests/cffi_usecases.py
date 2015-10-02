from __future__ import print_function, division, absolute_import

from numba import cffi_support


if cffi_support.SUPPORTED:
    from cffi import FFI
    defs = """
    double sin(double x);
    double cos(double x);
    """

    # Create inline module

    ffi = FFI()
    ffi.cdef(defs)
    C = ffi.dlopen(None) # loads the entire C namespace
    cffi_sin = C.sin
    cffi_cos = C.cos

    # Compile out-of-line module and load it

    ffi_ool = FFI()
    ffi.set_source('cffi_usecases_ool', defs)
    ffi.cdef(defs, override=True)
    ffi.compile()

    import cffi_usecases_ool
    cffi_support.register_module(cffi_usecases_ool)
    cffi_sin_ool = cffi_usecases_ool.lib.sin
    cffi_cos_ool = cffi_usecases_ool.lib.cos


def use_cffi_sin(x):
    return cffi_sin(x) * 2

def use_two_funcs(x):
    return cffi_sin(x) - cffi_cos(x)

def use_cffi_sin_ool(x):
    return cffi_sin_ool(x) * 2

def use_two_funcs_ool(x):
    return cffi_sin_ool(x) - cffi_cos_ool(x)

def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)
