#
# This show-cases a rather simple use of OpenCL
#

import ctypes
from numba.ocl.ocldrv.driver import cl


MEM_SIZE=128
out = (ctypes.c_char*MEM_SIZE)()

opencl_program = b"""
__kernel void hello(__global char* string)
{
    string[0] = 'H';
    string[1] = 'e';
    string[2] = 'l';
    string[3] = 'l';
    string[4] = 'o';
    string[5] = ',';
    string[6] = ' ';
    string[7] = 'W';
    string[8] = 'o';
    string[9] = 'r';
    string[10] = 'l';
    string[11] = 'd';
    string[12] = '!';
    string[13] = '\\0';
}
"""

ctxt = cl.create_context()
use_device = cl.platforms[-1].all_devices[-1]
print("Using device:\n{0}".format(use_device))
q = ctxt.create_command_queue(use_device)
mem = ctxt.create_buffer(MEM_SIZE)
program = ctxt.create_program_from_source(opencl_program)
program.build()
kernel = program.create_kernel(b"hello")
kernel.set_arg(0, mem)
q.enqueue_task(kernel)
q.enqueue_read_buffer(mem, 0, MEM_SIZE, out)
q.flush()
q.finish()

print(out.value)
