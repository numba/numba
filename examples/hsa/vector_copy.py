"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

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
    aql.dimensions = len(src.shape)
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
    hsa.hsa_agent_iterate_regions(gpu._id, get_kerarg, ctypes.byref(kernarg_region))
    assert kernarg_region != 0

    kernel_arg_buffer_size = code_descriptor._id.kerarg_segment_byte_size
    kernel_arg_buffer = ctypes.c_void_p()
    hsa.hsa_memory_allocate(kernarg_region, kernel_arg_buffer_size,
                            ctypes.byref(kernel_arg_buff))
    kernargs = ctypes.cast(kernel_arg_buffer, ctypes.c_void_p*2)
    kernargs[0] = dst.ctypes.data
    kernargs[1] = src.ctypes.data

    aql.kernel_object_address = code_descriptor._id.code_handle
    aql.kernarg_address = kernel_arg_buffer.value

    # going further requires modifying the API handling



if __name__=='__main__':
    src = np.random.random(1024*1024)
    dst = np.empty_like(src)
    main(src, dst)
    if np.array_equal(src, dst):
        print("PASSED")
    else:
        print("FAILED")
