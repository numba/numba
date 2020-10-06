from __future__ import absolute_import, print_function
import llvmlite.binding as ll
import os
from ctypes.util import find_library


def init_jit():
    from numba.dppl.dispatcher import DPPLDispatcher
    return DPPLDispatcher

def initialize_all():
    from numba.core.registry import dispatcher_registry
    dispatcher_registry.ondemand['dppl'] = init_jit

    import dpctl
    import glob
    import platform as plt
    platform = plt.system()
    if platform == 'Windows':
        paths = glob.glob(os.path.join(os.path.dirname(dpctl.__file__), '*DPPLSyclInterface.dll'))
    else:
        paths = glob.glob(os.path.join(os.path.dirname(dpctl.__file__), '*DPPLSyclInterface*'))

    if len(paths) == 1:
        ll.load_library_permanently(find_library(paths[0]))
    else:
        raise ImportError

    ll.load_library_permanently(find_library('OpenCL'))
