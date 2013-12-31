from __future__ import print_function
from pprint import pprint
import numpy as np
from numba.compiler import compile_isolated
from numba.tests import usecases
from numba import types
from timeit import default_timer as timer


def main():
    pyfunc = usecases.copy_arrays2d
    arraytype = types.Array(types.int32, 2, 'A')
    ctx, cfunc = compile_isolated(pyfunc, (arraytype, arraytype))

    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    b = np.empty_like(a)

    cfunc(a, b)

    print(b)



if __name__ == '__main__':
    main()
