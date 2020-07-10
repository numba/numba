from __future__ import absolute_import, print_function
import llvmlite.binding as ll
import os


def init_jit():
    from numba.dppl.dispatcher import DPPLDispatcher
    return DPPLDispatcher

def initialize_all():
    from numba.core.registry import dispatcher_registry
    dispatcher_registry.ondemand['dppl'] = init_jit

    ll.load_library_permanently('libdpglue.so')
    ll.load_library_permanently('libOpenCL.so')
