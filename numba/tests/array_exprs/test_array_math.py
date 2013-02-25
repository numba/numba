from numba import error
import numba
from numba import *
import numpy as np

def get_functions():
    def sqrt(a):
        return 2.7 + np.sqrt(a) + 1.6

    def log(a):
        return 2.7 + np.log(a) + 1.6

    def log10(a):
        return 2.7 + np.log10(a) + 1.6

    def log1p(a):
        return 2.7 + np.log1p(a) + 1.6

    def log2(a):
        return 2.7 + np.log2(a) + 1.6

    def exp(a):
        return 2.7 + np.exp(a) + 1.6

#    def expm1(a):
#        return 2.7 + np.expm1(a) + 1.6

    def sin(a):
        return 2.7 + np.sin(a) + 1.6

    def cos(a):
        return 2.7 + np.cos(a) + 1.6

    def absolute(a):
        return 2.7 + np.abs(a) + 1.6

    return locals()

dest_types = [int_, short, float_, double] #, complex128]

def test_math_funcs():
    functions = get_functions()
    exceptions = 0
    for func_name in functions:
        # func_name = 'sqrt'
        func = functions[func_name]
        for dest_type in dest_types:
            dest_type = dest_type[:, :]
            signature = dest_type(dest_type)

            try:
                numba_func = jit(signature)(func)
            except error.NumbaError, e:
                exceptions += 1
                print func_name, dest_type, e
                continue

            dtype = dest_type.dtype.get_dtype()
            x = np.arange(8 * 12, dtype=dtype).reshape(8, 12)
            x = ((x + 10) / 5).astype(dtype)

            r1 = numba_func(x)
            r2 = func(x)
            assert np.allclose(r1, r2), (r1, r2, signature, func_name)

    if exceptions:
        raise Exception

if __name__ == '__main__':
#    x = np.arange(8 * 12, dtype=np.int64).reshape(8, 12)
#    x = (10.0 / (x + 1)).astype(np.int64)
#    @autojit
#    def sqrt(a):
#        return 2.7 + np.cos(a) + 1.6
#    print sqrt.py_func(x) - sqrt(x)
    test_math_funcs()
#    numba.nose_run()
