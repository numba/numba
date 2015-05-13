from __future__ import print_function, absolute_import, division

from numba import config

if config.ENABLE_CUDASIM:
    from .simulator_init import *
else:
    from .device_init import *
    from .device_init import _auto_device


def test():
    if not is_available():
        raise cuda_error()

    from .tests.cudapy.runtests import test as test_cudapy
    from .tests.cudadrv.runtests import test as test_cudadrv

    testseq = [("cudadrv", test_cudadrv),
               ("cudapy", test_cudapy)]

    for name, udt in testseq:
        print("Running", name)
        if not udt():
            print("Test failed", name)
            return False

    return True
