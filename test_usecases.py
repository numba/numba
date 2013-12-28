from __future__ import print_function
from pprint import pprint
from numba.compiler import compile_isolated
from numba.tests import usecases
from numba import types
from timeit import default_timer as timer

def main():
    pyfunc = usecases.andor
    ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32))
    args = 1, 2

    ts = timer()
    pyfunc(*args)
    te = timer()
    print(te - ts)

    ts = timer()
    cfunc(*args)
    te = timer()
    print(te - ts)




if __name__ == '__main__':
    main()
