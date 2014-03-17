from __future__ import print_function, absolute_import, division

# Re export
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, syncthreads,
                    shared, local, const, grid, atomic)

from .decorators import jit, autojit, declare_device
from .api import *


def test():
    from .tests.cudapy.runtests import test as test_cudapy
    from .tests.cudadrv.runtests import test as test_cudadrv

    testseq = [("cudadrv", test_cudadrv),
               ("cudapy", test_cudapy)]
    for name, udt in testseq:
        print("Running", name)
        if not udt():
            print("Test failed", name)


