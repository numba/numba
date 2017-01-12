from __future__ import absolute_import, print_function

import numpy as np

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
    wavebarrier,
    activelanepermute_wavewidth,
)

from .decorators import (
    jit,
)

from .enums import (
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE
)

from .hsadrv.driver import hsa as _hsadrv


class _AutoDeregister(object):
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        deregister(*self.args)


def register(*args):
    """Register data into the HSA system

    Returns a contextmanager for use in with-context for auto deregistration.

    Use in context:

        with hsa.register(array):
            do_work_on_HSA(array)

    """
    for data in args:
        if isinstance(data, np.ndarray):
            _hsadrv.hsa_memory_register(data.ctypes.data, data.nbytes)
        else:
            raise TypeError(type(data))
    return _AutoDeregister(args)


def deregister(*args):
    """Deregister data form the HSA system
    """
    for data in args:
        if isinstance(data, np.ndarray):
            _hsadrv.hsa_memory_deregister(data.ctypes.data, data.nbytes)
        else:
            raise TypeError(type(data))
