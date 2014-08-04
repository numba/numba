from __future__ import print_function, absolute_import, division

from .stubs import (get_global_id, get_local_id, get_global_size,
                    get_local_size)
from .ocldrv.error import OpenCLSupportError

_exported = """
to_device
jit
"""


def _dummy(*args, **kws):
    """
    A dummy function that will supplant the exported API functions when OCL is not available.
    """
    raise ocl_error


try:
    from .api import *

    is_available = True
    ocl_error = None
except OpenCLSupportError as e:
    is_available = False
    ocl_error = e
    for name in _exported.split():
        globals()[name] = _dummy

del (_dummy)


def test():
    if not is_available:
        raise ocl_error

    from .tests.ocldrv.runtests import test as test_ocl_drv
    from .tests.oclpy.runtests import test as test_ocl_py

    testseq = [("ocldrv", test_ocl_drv),
               ("oclpy", test_ocl_py)]

    for name, udt in testseq:
        print("Running ", name)
        if not udt():
            print("Test failed ", name)


def _initialize():
    from numba.npyufunc import Vectorize

    def init_vectorize():
        from numba.ocl.ufunc import OclVectorize

        return OclVectorize

    Vectorize.target_registry.ondemand['ocl'] = init_vectorize


_initialize()
