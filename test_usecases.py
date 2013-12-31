from __future__ import print_function
from pprint import pprint
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba.tests import usecases
from numba import types
from timeit import default_timer as timer


def main():
    pyfunc = usecases.string1
    flags = Flags()
    flags.set("enable_pyobject")
    ctx, cfunc = compile_isolated(pyfunc, (types.int32, types.int32),
                                  flags=flags)
    a = 123
    b = 321

    ts = timer()
    r = pyfunc(a, b)
    te = timer()
    print(te - ts, r)

    ts = timer()
    r = cfunc(a, b)
    te = timer()
    print(te - ts, r)


if __name__ == '__main__':
    main()
