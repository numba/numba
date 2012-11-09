import inspect

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

    def exp(a, b):
        result = a**2 + b**2
        return np.exp(result) + 1.6

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

dest_types = [float_, double] # , complex64, complex128]

def test_math_funcs():
    functions = get_functions()
    exceptions = 0
    for func_name, func in functions.iteritems():
        # func = functions['absolute']
        for dest_type in dest_types:
            signature = dest_type(dest_type, dest_type)

            try:
                numba_func = jit(signature)(func)
            except error.NumbaError, e:
                exceptions += 1
                print func_name, dest_type, e
                continue

            r1 = numba_func(5.2, 6.9)
            r2 = func(5.2, 6.9)
            assert np.allclose(r1, r2), (r1, r2, dest_type, func_name)

    if exceptions:
        raise Exception

if __name__ == "__main__":
    test_math_funcs()