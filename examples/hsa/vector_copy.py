"""
Sample low-level HSA runtime example.
"""
from __future__ import print_function, division

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule



def main():
    WIDTH = 4
    HEIGHT = 4
    # note that hsa library is automaticaaly initialized on first use
    components = [a for a in hsa.agents if a.is_component]

    # select the first one
    if len(components) < 1:
        sys.exit("No HSA component found!")

    gpu = components[0]

    print("Using agent: {0} with queue size: {1}".format(gpu.name, gpu.queue_max_size))
    q = gpu.create_queue_multi(gpu.queue_max_size)

    a = np.random.random(WIDTH*HEIGHT).reshape((HEIGHT, WIDTH))
    b = np.random.random(WIDTH*HEIGHT).reshape((WIDTH, HEIGHT))
    print("input matrix A:\n", a)
    print("input matrix B:\n", b)

    # load Brig
    brig_module = BrigModule.from_file('vector_copy.brig')
    print ("Module created: ", brig_module)

    #program = hsa.create_program([gpu])
    #module = program.add_module(module)

    # finalize
    #program.finalize([gpu], '&__vector_copy_kernel')

    s = hsa.create_signal(1)



if __name__=='__main__':
    main()
