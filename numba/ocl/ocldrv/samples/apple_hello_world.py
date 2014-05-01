#
# A Python version of Apple's OpenCL Hello World.
# Original C source code can be seen at:
# https://developer.apple.com/library/mac/samplecode/OpenCL_Hello_World_Example/Listings/hello_c.html

from numba.ocl.ocldrv.driver import driver as cl
import numpy as np


DATA_SIZE = 1024

opencl_source = """
__kernel void square(__global float* input, __global float* output, const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        output[i] = input[i] * input[i];
}
"""

data = np.random.random_sample(DATA_SIZE).astype(np.float32)
platform = cl.platforms[0]
device = platform.devices[-1]

ctxt = cl.create_context(platform, [device])
q = ctxt.create_command_queue(device)
program = ctxt.create_program_from_source(opencl_source)
program.build()
kernel = program.create_kernel(b"square")
input = ctxt.create_buffer(len(data.data))
output = ctxt.create_buffer(len(data.data))
q.enqueue_write_buffer(input, 0, len(data.data), data.ctypes.data)
kernel.set_arg(0, input)
kernel.set_arg(1, output)
kernel.set_arg(2, DATA_SIZE)
local_sz = kernel.get_work_group_size_for_device(device)
global_sz = DATA_SIZE
q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
q.finish()
results = np.empty_like(data)
q.enqueue_read_buffer(output, 0, len(results.data), results.ctypes.data)
print("Computed '{0}/{1}' correct values.".format(np.sum(results == data*data), DATA_SIZE))
