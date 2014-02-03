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

dtypes = ['i', 'l', 'f', 'd', np.complex128]

def test_math_funcs():
    functions = get_functions()
    exceptions = 0
    for func_name in functions:
        # func_name = 'sqrt'
        func = functions[func_name]
        for dtype in dtypes:
            numba_func = autojit(func)

            x = np.arange(8 * 12, dtype=dtype).reshape(8, 12)
            x = ((x + 10) / 5).astype(dtype)

            r1 = numba_func(x)
            r2 = numba_func.py_func(x)
            assert np.allclose(r1, r2), (r1 - r2, r1.dtype, r2.dtype,
                                         func_name, x.dtype)

    if exceptions:
        raise Exception

if __name__ == '__main__':
    test_math_funcs()
