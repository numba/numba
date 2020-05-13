from __future__ import absolute_import, print_function
import llvmlite.binding as ll
import os
from distutils import sysconfig


def init_jit():
    from numba.dppy.dispatcher import DPPyDispatcher
    return DPPyDispatcher

def initialize_all():
    from numba.core.registry import dispatcher_registry
    dispatcher_registry.ondemand['dppy'] = init_jit

    ll.load_library_permanently('libdpglue.so')
    ll.load_library_permanently('libOpenCL.so')
