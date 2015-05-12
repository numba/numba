from __future__ import print_function, absolute_import, division

# Re export
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, syncthreads,
                    shared, local, const, grid, gridsize, atomic)
from .cudadrv.error import CudaSupportError
from . import initialize
from .errors import KernelRuntimeError

from .decorators import jit, autojit, declare_device
from .api import *
from .api import _auto_device


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.

    This will initialize the driver if it hasn't been initialized.
    """
    return driver.driver.is_available


def cuda_error():
    """Returns None or an exception if the CUDA driver fails to initialize.
    """
    return driver.driver.initialization_error

initialize.initialize_all()
