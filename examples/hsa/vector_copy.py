"""
Sample low-level HSA runtime example.

This sample tries to mimick the vector_copy example
"""
from __future__ import print_function, division

import numpy as np
from numba.hsa.hsadrv.driver import hsa, BrigModule


def create_program(device, brig_file, symbol):
    brig_module = BrigModule.from_file(brig_file)
    symbol_offset = brig_module.find_symbol_offset(symbol)
    print("symbol {0} at offset {1}".format(symbol, symbol_offset))
    program = hsa.create_program([device])
    module = program.add_module(brig_module)

    program.finalize(device, module, symbol_offset)

    return program

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

    program = create_program(gpu, 'vector_copy.brig', '&__vector_copy_kernel')
    print(program)

    s = hsa.create_signal(1)



if __name__=='__main__':
    main()
