#!/usr/bin/env python
# Demonstration of using MPI and Numba CUDA to perform parallel computation
# using GPUs in multiple nodes. This example requires MPI4py to be installed.
#
# The root process creates an input data array that is scattered to all nodes.
# Each node calls a CUDA jitted function on its portion of the input data.
# Output data is then gathered back to the master node.
#
# Notes/limitations:
#
# 1. It is generally more efficient to avoid initialising all data on the root
#    node then scattering it out to all other nodes, and instead each node
#    should initialise its own data, but initialisation is done on the root node
#    here to keep the example simple.
# 2. If multiple GPUs are available to a single MPI process, additional code may
#    need adding to ensure the correct GPU is used by each process - this will
#    depend on the exact configuration of the MPI cluster.
#
# This example can be invoked with:
#
#     $ mpirun -np <np> python cuda_mpi.py
#
# where np is the number of processes (e.g. 4). For demonstrating the code, this
# does work with a single node and a single GPU, since multiple processes can
# share a single GPU. However, in a production setting, it may be more
# appropriate to provide one GPU per MPI process.

from __future__ import print_function

from mpi4py import MPI
from numba import cuda
import numpy as np

mpi_comm = MPI.COMM_WORLD

# Input data size
total_n = 10


# Process 0 creates input data
if mpi_comm.rank == 0:
    input_data = np.arange(total_n, dtype=np.int32)
    print("Input:", input_data)
else:
    input_data = None


# Compute partitioning of the input array
proc_n = [ total_n // mpi_comm.size + (total_n % mpi_comm.size > n)
                   for n in range(mpi_comm.size) ]
pos = 0
pos_n = []
for n in range(mpi_comm.size):
    pos_n.append(pos)
    pos += proc_n[n]

my_n = proc_n[mpi_comm.rank]
my_offset = pos_n[mpi_comm.rank]
print('Process %d, my_n = %d' % (mpi_comm.rank, my_n))
print('Process %d, my_offset = %d' % (mpi_comm.rank, my_offset))


# Distribute input data across processes
my_input_data = np.zeros(my_n, dtype=np.int32)
mpi_comm.Scatterv([input_data, proc_n, pos_n, MPI.INT], my_input_data)
print('Process %d, my_input_data = %s' % (mpi_comm.rank, my_input_data))


# Perform computation on local data

@cuda.jit
def sqplus2(input_data, output_data):
    for i in range(len(input_data)):
        d = input_data[i]
        output_data[i] = d * d + 2


my_output_data = np.empty_like(my_input_data)
sqplus2(my_input_data, my_output_data)
print('Process %d, my_output_data = %s' % (mpi_comm.rank, my_output_data))


# Bring result back to root process
if mpi_comm.rank == 0:
    output_data = np.empty_like(input_data)
else:
    output_data = None

mpi_comm.Gatherv(my_output_data, [output_data, proc_n, pos_n, MPI.INT])

if mpi_comm.rank == 0:
    print("Output:", output_data)


MPI.Finalize()
