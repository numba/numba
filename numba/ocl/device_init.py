from __future__ import print_function, absolute_import, division

# Re export
from .stubs import (
    get_global_id,
    get_global_size,
    get_local_id,
    get_local_size,
    get_group_id,
    get_work_dim,
    get_num_groups,
    barrier,
    mem_fence,
    shared,
    sub_group_barrier,
)
from .ocldrv.error import OclSupportError
#from .ocldrv import nvvm
from . import initialize
from .errors import KernelRuntimeError

from .decorators import jit, autojit, declare_device
from .api import *
from .api import _auto_device


def is_available():
    """Returns a boolean to indicate the availability of a OpenCL GPU.

    This will initialize the driver if it hasn't been initialized.
    """
    return driver.driver.is_available and nvvm.is_available()


def ocl_error():
    """Returns None or an exception if the OpenCL driver fails to initialize.
    """
    return driver.driver.initialization_error

initialize.initialize_all()
