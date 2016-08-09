from __future__ import absolute_import, division, print_function

import multiprocessing as mp

import numpy as np

from numba import cuda


def parent():
    arr = np.arange(10)
    darr = cuda.to_device(arr)
    ipch = darr.get_ipc_handle()

    # launch child proc
    mpc = mp.get_context('spawn')
    queue = mpc.Queue()
    childproc = mpc.Process(target=child, args=[queue])

    childproc.start()
    queue.put(ipch)
    childproc.join(1)
    hostarr = queue.get()

    print('original array:', arr)
    # device array is modified by child process
    print('device array:', darr.copy_to_host())
    print('returned host array', hostarr)

    # verify
    np.testing.assert_equal(darr.copy_to_host(), arr + 1)
    np.testing.assert_equal(hostarr, arr * 2)


@cuda.jit
def plus1(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] += 1


def child(queue):
    ipch = queue.get()
    with ipch as darr:
        # keep a copy
        arr = darr.copy_to_host()
        # modify host array
        arr *= 2
        # modify device array directly
        plus1[(darr.size + 64 - 1) // 64, 64](darr)
    # send host array back
    queue.put(arr)


def main():
    parent()


if __name__ == '__main__':
    main()
