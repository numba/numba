from __future__ import absolute_import, print_function, division

import multiprocessing as mp
import traceback

import numpy as np

from numba import cuda
from numba.cuda.cudadrv import drvapi, devicearray
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase


def base_ipc_handle_test(handle_array, size, result_queue):
    try:
        # manually recreate the IPC mem handle
        handle = drvapi.cu_ipc_mem_handle()
        for i, v in enumerate(handle_array):
            handle[i] = v
        # use *IpcHandle* to open the IPC memory
        handle = cuda.driver.IpcHandle(None, handle, size)
        dptr = handle.open(cuda.current_context())
        # read the device pointer as an array
        dtype = np.dtype(np.intp)
        darr = devicearray.DeviceNDArray(shape=(size // dtype.itemsize,),
                                         strides=(dtype.itemsize,),
                                         dtype=dtype,
                                         gpu_data=dptr)
        # copy the data to host
        arr = darr.copy_to_host()
        del darr, dptr
        handle.close()
    except:
        # FAILED. propagate the exception as a string
        succ = False
        out = traceback.format_exc()
    else:
        # OK. send the ndarray back
        succ = True
        out = arr
    result_queue.put((succ, out))


@skip_on_cudasim('Ipc not available in CUDASIM')
class TestIpcMemory(CUDATestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method('spawn')

    def test_ipc_handle(self):
        # prepare data for IPC
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)

        # create IPC handle
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)

        # manually prepare for serialization (as ndarray)
        handle_array = np.asarray(ipch.handle)
        size = ipch.size

        # spawn new process for testing
        result_queue = mp.Queue()
        args = (handle_array, size, result_queue)
        proc = mp.Process(target=base_ipc_handle_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(arr, out)
        proc.join(3)


if __name__ == '__main__':
    unittest.main()