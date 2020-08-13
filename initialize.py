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

    ll.load_library_permanently(find_library('DPPLOpenCLInterface'))
    ll.load_library_permanently(find_library('OpenCL'))
