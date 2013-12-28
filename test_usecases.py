from __future__ import print_function
from pprint import pprint
from numba.compiler import compile_isolated
from numba.tests import usecases
from numba import types


def main():
    pyfunc = usecases.andor
    ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))
    args = 1, 2
    print(pyfunc(*args))
    print(cfunc(*args))




if __name__ == '__main__':
    main()
