"""
An example that demonstrate the CUDA features with
dask <http://dask.pydata.org/> using the "Bag" collections.

The script can be configured to use multiprocessing or multithreading.
"""

from __future__ import print_function, division, absolute_import

import sys
import math
import numpy as np
import dask
import dask.bag as db
import dask.threaded
import dask.multiprocessing

from numba import vectorize, cuda, float32


@vectorize(['float32(float32)'], target='cuda')
def gpu_cos(x):
    """
    A simple CUDA ufunc to compute the elemwise cosine
    """
    return math.cos(x)


# maximum size blockdim of ``gpu_single_block_sum`` kernel.
# also, the shared memory size of the kernel
gpu_block_sum_max_blockdim = 512


@cuda.jit
def gpu_single_block_sum(arr, out):
    """
    A naive single threadblock sum reduction
    """
    temp = cuda.shared.array(gpu_block_sum_max_blockdim, dtype=float32)
    tid = cuda.threadIdx.x
    blksz = cuda.blockDim.x
    temp[tid] = 0
    # block stride loop to sum-reduce cooperatively
    for i in range(tid, arr.size, blksz):
        temp[tid] += arr[i]
    cuda.syncthreads()
    # naive intra block sum that uses a single thread
    if tid == 0:
        for i in range(1, blksz):
            temp[0] += temp[i]
        # store result
        out[0] = temp[0]


def sum_parts(data):
    """
    Driver for ``gpu_single_block_sum`` kernel
    """
    arr = np.asarray(data, dtype=np.float32)
    out = cuda.device_array(1, dtype=np.float32)
    gpu_single_block_sum[1, gpu_block_sum_max_blockdim](arr, out)
    return out.copy_to_host()[0]


def main(kind):
    input_array = np.random.random(5000)

    getter = {'processes': dask.multiprocessing.get,
              'threads': dask.threaded.get}[kind]

    # sets the scheduler
    with dask.set_options(get=getter):

        # set ``partition_size`` to ensure each partition has enough work
        bag = db.from_sequence(input_array, partition_size=1000)

        # compute elemwise cosine on the gpu within each partition
        bag_cos = bag.map_partitions(
            lambda x: gpu_cos(np.asarray(x, dtype=np.float32)))

        # apply partial sum-reduce on each partition
        # then, finish it on the host
        got = bag_cos.reduction(sum_parts, sum).compute()

        # cross validate with numpy
        expected = np.sum(np.cos(input_array))

        print('Got:     ', got)
        print('Expected:', expected)
        correct = np.allclose(got, expected)
        print('Correct: ', correct)
        sys.exit(0 if correct else 1)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 1:
        main(argv[0])
    else:
        print('''
Usage: {name} <scheduler>

Args:
    scheduler: dask scheduler to use; either "processes" or "threads"

'''.format(name=sys.argv[0]))
