"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

import sys
import ctypes

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule
from numba.hsa.hsadrv import drvapi, enums

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

    program, code_descriptor = create_program(gpu, 'vector_copy.brig', '&__vector_copy_kernel')
    print(program)

    s = hsa.create_signal(1)

    # manually build an aql packet
    aql = drvapi.hsa_dispatch_packet_t()

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

    kernargs = ctypes.POINTER(ctypes.c_void_p*2)
    hsa.hsa_memory_allocate(kernarg_region, kernel_arg_buffer_size,
                            ctypes.byref(kernargs))
    kernargs.contents[0] = dst.ctypes.data
    kernargs.contents[1] = src.ctypes.data

    hsa.hsa_memory_register(kernargs, ctypes.sizeof(kernargs))

    aql.kernel_object_address = code_descriptor._id.code.handle
    aql.kernarg_address = kernel_arg_buffer.value

    index = hsa.hsa_queue_load_write_index_relaxed(q._id)
    print ('using slot in queue: {0}'.format(index))
    queueMask = q._id.contents.size - 1
    real_index = index & queueMask
    packet_array = ctypes.cast(q._id.contents.base_address,
                               ctypes.POINTER(drvapi.hsa_dispatch_packet_t))
    packet_array[real_index] = aql
    hsa.hsa_queue_store_write_index_relaxed(q._id, index+1)
    hsa.hsa_signal_store_relaxed(q.doorbell_signal, index)

    print ('wait for the signal to be raised')
    # wait for results
    hsa.hsa_signal_wait_acquire(s._id, enums.HSA_LT, 1, -1, enums.HSA_WAIT_EXPECTANCY_UNKNOWN)

    hsa.hsa_memory_free(kernel_arg_buffer)


if __name__=='__main__':
    src = np.random.random(1024*1024).astype(np.float32)
    dst = np.zeros_like(src)
    main(src, dst)
    print(src, dst)
    if np.array_equal(src, dst):
        print("PASSED")
    else:
        print("FAILED")
