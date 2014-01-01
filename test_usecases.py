from __future__ import print_function
from pprint import pprint
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba.tests import usecases
from numba import types
from timeit import default_timer as timer


def main():
    pyfunc = usecases.blackscholes_cnd
    flags = Flags()
    # flags.set("enable_pyobject")
    ctx, cfunc, err = compile_isolated(pyfunc, (types.float32,),
                                       flags=flags)

    d = 0.5
    args = d,

    ts = timer()
    r = pyfunc(*args)
    te = timer()
    pytime = te - ts
    print(pytime, r)

    ts = timer()
    r = cfunc(*args)
    te = timer()
    jittime = te - ts
    print(jittime, r)

    print(pytime/jittime)


if __name__ == '__main__':
    main()
