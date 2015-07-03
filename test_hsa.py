import os.path
import ctypes
import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule, Executable, Program

agent = list(hsa.agents)[1]
print(agent)
print('queue_max_size', agent.queue_max_size)
queue = agent.create_queue_single(2 ** 5)

brig = BrigModule.from_file(os.path.join('numba',
                                         'hsa',
                                         'tests',
                                         'hsadrv',
                                         'vector_copy.brig'))

program = Program()
print(program)

program.add_module(brig)
print('isa', hex(agent.isa))

code = program.finalize(agent.isa)
del program
ex = Executable()
ex.load(agent, code)
ex.freeze()

sym = ex.get_symbol(agent, "&__vector_copy_kernel")
print(sym.kernel_object)
print(sym.kernarg_segment_size)
print(sym.group_segment_size)
print(sym.private_segment_size)

sig = hsa.create_signal(1)

kernarg_region = [r for r in agent.regions if r.supports_kernargs][0]

kernarg_types = (ctypes.c_void_p * 2)
kernargs = kernarg_region.allocate(kernarg_types)

src = np.random.random(1024 * 1024).astype(np.float32)
dst = np.zeros_like(src)

kernargs[0] = src.ctypes.data
kernargs[1] = dst.ctypes.data


hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
hsa.hsa_memory_register(dst.ctypes.data, dst.nbytes)
hsa.hsa_memory_register(ctypes.byref(kernargs), ctypes.sizeof(kernargs))


# queue.dispatch(code_descriptor, kernargs, workgroup_size=(256,), grid_size=(
#     1024*1024,))
