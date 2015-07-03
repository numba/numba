import os.path
from numba.hsa.hsadrv.driver import hsa, BrigModule, Executable

agent = list(hsa.agents)[1]
print(agent)
print('queue_max_size', agent.queue_max_size)
agent.create_queue_single(2 ** 5)

brig = BrigModule.from_file(os.path.join('numba',
                                         'hsa',
                                         'tests',
                                         'hsadrv',
                                         'vector_copy.brig'))

program = hsa.create_program()
print(program)

program.add_module(brig)
print('isa', hex(agent.isa))

code = program.finalize(agent.isa)

ex = Executable()

ex.load(agent, code)

ex.freeze()

sym = ex.get_symbol(agent, "&__vector_copy_kernel")

print(sym.kernel_object)
print(sym.kernarg_segment_size)
print(sym.group_segment_size)
print(sym.private_segment_size)
