from __future__ import print_function, absolute_import, division
import functools

# Re export
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, syncthreads,
                    shared, local, const, grid, atomic)
from .cudadrv.error import CudaSupportError


def dummy(*args, **kws):
    """A dummy function to allow import of CUDA functions even if the CUDA
    is not available.
    """
    raise cuda_error


_exported = """
jit
autojit
declare_device
require_context
current_context
to_device
device_array
pinned_array
mapped_array
synchronize
device_array_like
stream
pinned
mapped
event
select_device
get_current_device
list_devices
close
detect
defer_cleanup
"""

try:
    from .decorators import jit, autojit, declare_device
    from .api import *

    is_available = True
    cuda_error = None
except CudaSupportError as e:
    is_available = False
    cuda_error = e
    for name in _exported.split():
        globals()[name] = dummy


def test():
    if not is_available:
        raise cuda_error

    from .tests.cudapy.runtests import test as test_cudapy
    from .tests.cudadrv.runtests import test as test_cudadrv

    testseq = [("cudadrv", test_cudadrv),
               ("cudapy", test_cudapy)]

    for name, udt in testseq:
        print("Running", name)
        if not udt():
            print("Test failed", name)

