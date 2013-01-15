import inspect

from numba.minivect import minitypes
from numba import error
import numba
from numba import *
import numpy as np

def get_functions():
    def sqrt(a, b):
        result = a**2 + b**2
        return np.sqrt(result) + 1.6

    def log(a, b):
        result = a**2 + b**2
        return np.log(result) + 1.6

    def log10(a, b):
        result = a**2 + b**2
        return np.log10(result) + 1.6

    def log1p(a, b):
        result = a**2 + b**2
        return np.log1p(result) + 1.6

    def log2(a, b):
        result = a**2 + b**2
        return np.log2(result) + 1.6

    def exp(a, b):
        result = a**2 + b**2
        return np.exp(result) + 1.6

    def expm1(a, b):
        result = a**2 + b**2
        return np.expm1(result) + 1.6

    def sin(a, b):
        result = a**2 + b**2
        return np.sin(result) + 1.6

    def cos(a, b):
        result = a**2 + b**2
        return np.cos(result) + 1.6

    def absolute(a, b):
        result = a**2 + b**2
        return np.abs(result) + 1.6

    return locals()

dest_types = [int_, short, Py_ssize_t, float_, double, complex128]

def test_math_funcs():
    functions = get_functions()
    exceptions = 0
    for func_name in functions:
        # func_name = 'sqrt'
        func = functions[func_name]
        for dest_type in dest_types:
            signature = minitypes.FunctionType(None, [dest_type, dest_type])

            try:
                numba_func = jit(signature)(func)
            except error.NumbaError, e:
                exceptions += 1
                print func_name, dest_type, e
                continue

            x, y = 5.2, 6.9
            if dest_type.is_int:
                x, y = 5, 6

            r1 = numba_func(x, y)
            r2 = func(x, y)
            assert np.allclose(r1, r2), (r1, r2, signature, func_name)

    if exceptions:
        raise Exception

@autojit
def sin(A):
    return np.sin(A)

def test_array_math():
#    A = np.arange(10)
#    assert np.all(sin(A) == sin.py_func(A))
    dst_types = set(dest_types)
    dst_types.discard(Py_ssize_t)

    functions = get_functions()
    for func_name, func in functions.iteritems():
        for dst_type in dst_types:
            dtype = dst_type.get_dtype()
            a = np.arange(100, dtype=dtype)
            b = np.arange(50, 150, dtype=dtype)
            r1 = autojit(func)(a, b)
            r2 = func(a, b)
            assert np.allclose(r1, r2)


if __name__ == "__main__":
#    @jit(complex64(complex64, complex64))
#    def log1(a, b):
#        result = a**2 + b**2
#        return np.log(result) + 1.6
#
#    @jit(complex128(complex128, complex128))
#    def log2(a, b):
#        result = a**2 + b**2
#        return np.log(result) + 1.6
#
#    print log1(5.2, 6.9)
#    print log2(5.2, 6.9)
#    print log1.py_func(5.2, 6.9)
    test_array_math()
    test_math_funcs()