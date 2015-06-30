import os.path
from numba.hsa.hsadrv.driver import hsa, BrigModule

agent = list(hsa.agents)[1]
print(agent)
print('queue_max_size', agent.queue_max_size)
agent.create_queue_single(2 ** 5)

brig = BrigModule.from_file(os.path.join('numba',
                                         'hsa',
                                         'tests',
                                         'hsadrv',
                                         'vector_copy.brig'))

print(brig)


