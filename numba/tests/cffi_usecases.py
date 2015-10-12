from __future__ import print_function, division, absolute_import

from numba import cffi_support
from numba.tests.support import static_temp_directory
import sys
import numpy as np


if cffi_support.SUPPORTED:
    from cffi import FFI
    defs = """
    double sin(double x);
    double cos(double x);
    int foo(int a, int b, int c);
    """

    source = """
    static int foo(int a, int b, int c)
    {
        return a + b * c;
    }
    """

    # Create inline module

    ffi = FFI()
    ffi.cdef(defs)
    C = ffi.dlopen(None) # loads the entire C namespace
    cffi_sin = C.sin
    cffi_cos = C.cos

    # Compile out-of-line module and load it

    defs += """
    void vsSin(int n, float* x, float* y);
    void vdSin(int n, double* x, double* y);
    """

    source += """
    void vsSin(int n, float* x, float* y) {
        int i;
        for (i=0; i<n; i++)
            y[i] = sin(x[i]);
    }

    void vdSin(int n, double* x, double* y) {
        int i;
        for (i=0; i<n; i++)
            y[i] = sin(x[i]);
    }
    """

    ffi_ool = FFI()
    ffi.set_source('cffi_usecases_ool', source)
    ffi.cdef(defs, override=True)
    tmpdir = static_temp_directory('test_cffi')
    ffi.compile(tmpdir=tmpdir)
    sys.path.append(tmpdir)
    try:
        import cffi_usecases_ool
        cffi_support.register_module(cffi_usecases_ool)
        cffi_sin_ool = cffi_usecases_ool.lib.sin
        cffi_cos_ool = cffi_usecases_ool.lib.cos
        cffi_foo = cffi_usecases_ool.lib.foo
        vsSin = cffi_usecases_ool.lib.vsSin
        vdSin = cffi_usecases_ool.lib.vdSin
    finally:
        sys.path.remove(tmpdir)


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

def use_user_defined_symbols():
    return cffi_foo(1, 2, 3)

# The from_buffer method is member of cffi.FFI, and also of CompiledFFI objects
# (cffi_usecases_ool.ffi is a CompiledFFI object) so we use both in these
# functions.

def vector_sin_float32(x):
    y = np.empty_like(x)
    vsSin(len(x), ffi.from_buffer(x),
        cffi_usecases_ool.ffi.from_buffer(y))
    return y

def vector_sin_float64(x):
    y = np.empty_like(x)
    vdSin(len(x), ffi.from_buffer(x),
        cffi_usecases_ool.ffi.from_buffer(y))
    return y
