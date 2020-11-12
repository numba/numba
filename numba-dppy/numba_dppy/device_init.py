from __future__ import print_function, absolute_import, division

# Re export
from .ocl.stubs import (
    get_global_id,
    get_global_size,
    get_local_id,
    get_local_size,
    get_group_id,
    get_work_dim,
    get_num_groups,
    barrier,
    mem_fence,
    sub_group_barrier,
    atomic,
    local,
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE,
)

DEFAULT_LOCAL_SIZE = []

from . import initialize

from .decorators import kernel, func, autojit
import dpctl
from . import target


def is_available():
    """Returns a boolean to indicate the availability of a OpenCL GPU.

    This will initialize the driver if it hasn't been initialized.
    """
    return dpctl.has_gpu_queues()


#def ocl_error():
#    """Returns None or an exception if the OpenCL driver fails to initialize.
#    """
#    return driver.driver.initialization_error

initialize.initialize_all()
