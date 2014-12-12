"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

import sys
import os
import ctypes
from ctypes.util import find_library

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule
from numba.hsa.hsadrv import drvapi, enums

libc = ctypes.CDLL(find_library('c'))
libc.posix_memalign.restype = ctypes.c_int
libc.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t
]

libc.free.restype = None
libc.free.argtypes = [ctypes.c_void_p]


def dump_aql_packet(aql):
    for field_desc in drvapi.hsa_dispatch_packet_t._fields_:
        fname = field_desc[0]
        print (fname, getattr(aql, fname))

def alloc_aligned(typ, alignment):
    sz = ctypes.sizeof(typ)
    result = ctypes.c_void_p()
    err = libc.posix_memalign(ctypes.byref(result),
                              alignment, sz)
    if err != 0:
        raise Exception('ENOMEM')

    ctypes.memset(result, 0, sz)
    return ctypes.cast(result, ctypes.POINTER(typ)).contents

def free(obj):
    libc.free(ctypes.byref(obj))

def create_program(device, brig_file, symbol):
    brig_module = BrigModule.from_file(brig_file)
    symbol_offset = brig_module.find_symbol_offset(symbol)
    print("symbol {0} at offset {1}".format(symbol, symbol_offset))
    program = hsa.create_program([device])
    module = program.add_module(brig_module)

    code_descriptor = program.finalize(device, module, symbol_offset)

    return program, code_descriptor


def get_kernarg(region, result):
    flags = drvapi.hsa_region_flag_t()
    hsa.hsa_region_get_info(region, enums.HSA_REGION_INFO_FLAGS, ctypes.byref(flags))
    print("region {0} with flags {1}".format(hex(region), hex(flags.value)))
    if flags.value & enums.HSA_REGION_FLAG_KERNARG:
        result.value = region

    return enums.HSA_STATUS_SUCCESS


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

    s = hsa.create_signal(1)

    # manually build an aql packet
    aql = alloc_aligned(drvapi.hsa_dispatch_packet_t, 64)
    print (aql)

    aql.completion_signal = s._id
    aql.dimensions = 1
    aql.workgroup_size_x = 256
    aql.workgroup_size_y = 1
    aql.workgroup_size_z = 1
    aql.grid_size_x = 1024 * 1024
    aql.grid_size_y = 1
    aql.grid_size_z = 1
    aql.header.type = enums.HSA_PACKET_TYPE_DISPATCH
    aql.header.acquire_fence_scope = 2
    aql.header.release_fence_scope = 2
    aql.header.barrier = 1
    aql.group_segment_size = 0
    aql.private_segment_size = 0

    # setup kernel arguments
    hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
    hsa.hsa_memory_register(dst.ctypes.data, dst.nbytes)

    kernarg_region = drvapi.hsa_region_t(0)
    callback = drvapi.HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC(get_kernarg)
    hsa.hsa_agent_iterate_regions(gpu._id, callback, kernarg_region)
    assert kernarg_region != 0

    kernel_arg_buffer_size = code_descriptor._id.kernarg_segment_byte_size
    print ('Kernel has kernarg_segment_byte_size {0}'.format(kernel_arg_buffer_size))

    kernel_arg_buffer = ctypes.c_void_p()
    hsa.hsa_memory_allocate(kernarg_region, kernel_arg_buffer_size,
                            ctypes.byref(kernel_arg_buffer))
    kernargs = ctypes.cast(kernel_arg_buffer, ctypes.POINTER(ctypes.c_void_p * 2)).contents
    kernargs[0] = src.ctypes.data
    kernargs[1] = dst.ctypes.data

    print ('kernel_arg_buffer is {0}; kernargs is {1}'.format(hex(kernel_arg_buffer.value),
                                                              ctypes.byref(kernargs)))

    hsa.hsa_memory_register(ctypes.byref(kernargs), ctypes.sizeof(kernargs))

    aql.kernel_object_address = code_descriptor._id.code.handle
    aql.kernarg_address = kernel_arg_buffer.value

    dump_aql_packet(aql)

    print("pushing packet into the queue")
    index = hsa.hsa_queue_load_write_index_relaxed(q._id)
    print ('using slot in queue: {0}'.format(index))
    queueMask = q._id.contents.size - 1
    real_index = index & queueMask
    packet_array = ctypes.cast(q._id.contents.base_address,
                               ctypes.POINTER(drvapi.hsa_dispatch_packet_t))
    packet_array[real_index] = aql
    hsa.hsa_queue_store_write_index_relaxed(q._id, index+1)
    print("ringing the bell")
    hsa.hsa_signal_store_relaxed(q.doorbell_signal, index)

    print ('wait for the signal to be raised')
    # wait for results
    hsa.hsa_signal_wait_acquire(s._id, enums.HSA_LT, 1, -1, enums.HSA_WAIT_EXPECTANCY_UNKNOWN)

    hsa.hsa_memory_free(kernel_arg_buffer)
    free(aql)


if __name__=='__main__':
    src = np.random.random(1024*1024).astype(np.float32)
    dst = np.zeros_like(src)
    main(src, dst)
    print(src, dst)
    if np.array_equal(src, dst):
        print("PASSED")
    else:
        print("FAILED")
