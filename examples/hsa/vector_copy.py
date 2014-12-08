"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule


def create_program(brig_file, symbol):
    brig_module = BrigModule.from_file(brig_file)

    program = hsa.create_program([gpu])
    module_handle = program.add_module(brig_module)

def main():
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
