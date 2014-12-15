"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

import sys
import os
import ctypes
from time import time
from ctypes.util import find_library

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule
from numba.hsa.hsadrv import drvapi, enums


def dump_aql_packet(aql):
    for field_desc in drvapi.hsa_dispatch_packet_t._fields_:
        fname = field_desc[0]
        print (fname, getattr(aql, fname))


def create_program(device, brig_file, symbol):
    brig_module = BrigModule.from_file(brig_file)
    symbol_offset = brig_module.find_symbol_offset(symbol)
    print("symbol {0} at offset {1}".format(symbol, symbol_offset))
    program = hsa.create_program([device])
    module = program.add_module(brig_module)

    code_descriptor = program.finalize(device, module, symbol_offset)

    return program, code_descriptor


def main(src, dst):
    # note that the hsa library is automatically initialized on first use.
    # the list of agents is present in the driver object, so we can use
    # pythonic ways to enumerate/filter/select the agents/components we
    # want to use

    components = [a for a in hsa.agents if a.is_component]

    # select the first one
    if len(components) < 1:
        sys.exit("No HSA component found!")

    gpu = components[0]

    print("Using agent: {0} with queue size: {1}".format(gpu.name, gpu.queue_max_size))
    q = gpu.create_queue_multi(gpu.queue_max_size)

    basedir = os.path.dirname(__file__)
    brigfile = os.path.join(basedir, 'vector_copy.brig')
    if not os.path.isfile(brigfile):
        print("Missing brig file")
        sys.exit(1)

    program, code_descriptor = create_program(gpu, brigfile, '&__vector_copy_kernel')
    print(program)


    kernarg_regions = [r for r in gpu.regions if r.supports_kernargs]
    assert kernarg_regions
    kernarg_region = kernarg_regions[0]
    # Specify the argument types required by the kernel and allocate them
    # note: in an ideal world this should come from kernel metadata
    kernarg_types = ctypes.c_void_p * 2
    kernargs = kernarg_region.allocate(kernarg_types)
    kernargs[0] = src.ctypes.data
    kernargs[1] = dst.ctypes.data

    hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
    hsa.hsa_memory_register(dst.ctypes.data, dst.nbytes)
    hsa.hsa_memory_register(ctypes.byref(kernargs), ctypes.sizeof(kernargs))

    # sync (in fact, dispatch will create a dummy signal for the dispatch and
    # wait for it before returning)
    print("dispatch synchronous... ", end="")
    t_start = time()
    q.dispatch(code_descriptor, kernargs, workgroup_size=(256,), grid_size=(1024*1024,))
    t_end = time()
    print ("ellapsed: {0:10.9f} s.".format(t_end - t_start))

    # async: handle the signal by hand
    print("dispatch asynchronous... ", end="")
    t_start = time()
    s = hsa.create_signal(1)
    q.dispatch(code_descriptor, kernargs,
               workgroup_size=(256,), grid_size=(1024*1024,), signal=s)
    t_launched = time()
    hsa.hsa_signal_wait_acquire(s._id, enums.HSA_LT, 1, -1,
                                enums.HSA_WAIT_EXPECTANCY_UNKNOWN)
    t_end = time()
    print ("launch: {0:10.9f} s. total: {1:10.9g} s.".format(
        t_launched - t_start, t_end - t_start))

    # this is placed in the kernarg_region for symmetry, but shouldn't be required.
    kernarg_region.free(kernargs)


if __name__=='__main__':
    src = np.random.random(1024*1024).astype(np.float32)
    dst = np.zeros_like(src)
    main(src, dst)
    print(src, dst)
    if np.array_equal(src, dst):
        print("PASSED")
    else:
        print("FAILED")
