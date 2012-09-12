cimport utils
cimport cuda
cimport numpy as cnp

import os
import math
import logging

import numpy as np

cuda.init_cuda_exc_type()

cdef int get_device_number(int device_number):
    if device_number < 0:
        try:
            n = open(os.path.expanduser("~/.cuda_device")).read().strip()
        except IOError:
            pass
        else:
            try:
                device_number = int(n)
            except ValueError:
                logging.error("Invalid device in ~/.cuda_device: %s" % n)

    return device_number

cdef cuda.CUdevice get_device(int device_number) except *:
    cdef cuda.CUdevice result

    device_number = get_device_number(device_number)
    cuda.get_device(&result, NULL, device_number)
    return result

def compute_capability(int device_number):
    cdef cuda.CudaDeviceAttrs device_attrs
    cdef cuda.CUdevice cu_device = get_device(device_number)

    cuda.init_attributes(cu_device, &device_attrs)
    return (device_attrs.COMPUTE_CAPABILITY_MAJOR,
            device_attrs.COMPUTE_CAPABILITY_MINOR)

cdef class CudaFunction(utils.Function):
    cdef cuda.CUfunction cu_function
    cdef bytes func_name

    def __init__(self, func_name):
        self.func_name = func_name

    cdef load(self, cuda.CUmodule *cu_module):
        cuda.cuda_getfunc(cu_module[0], &self.cu_function, self.func_name)

    cdef int invoke(self, cnp.npy_intp *shape, char **data_pointers,
                    cnp.npy_intp **strides_pointers) nogil except -1:
        pass

cdef class CudaUFuncDispatcher(object): #cutils.UFuncDispatcher):

    cdef cuda.CudaDeviceAttrs device_attrs

    cdef cuda.CUdevice cu_device
    cdef cuda.CUcontext cu_context
    cdef cuda.CUmodule cu_module
    cdef cuda.CUfunction cu_ufunc

    cdef dict functions

    def __init__(self, ptx_code, types_to_name, device_number):
        cdef CudaFunction func

        cuda.get_device(&self.cu_device, &self.cu_context, device_number)
        cuda.init_attributes(self.cu_device, &self.device_attrs)
        cuda.cuda_load(ptx_code, &self.cu_module)

        self.functions = {}
        for dtypes, (result_dtype, name) in types_to_name.items():
            func = CudaFunction(name)
            func.load(&self.cu_module)
            self.functions[dtypes] = (result_dtype, func)

    def __call__(self, *args):
        cdef CudaFunction cuda_func
        cdef cnp.npy_intp N, MAX_THREAD, thread_count, block_count

        dtypes = tuple(a.dtype for a in args)
        if dtypes not in self.functions:
            raise TypeError("Input dtypes not supported by ufunc %s" % (dtypes,))

        result_dtype, cuda_func = self.functions[dtypes]

        # prepare broadcasted contiguous arrays
        # TODO: Allow strided memory (use mapped memory + strides?)
        # TODO: don't perform actual broadcasting, pass in strides
        args = [np.ascontiguousarray(a) for a in args]
        broadcast_arrays = np.broadcast_arrays(*args)
        N = np.prod(broadcast_arrays[0].shape)

        out = np.empty_like(broadcast_arrays[0], dtype=result_dtype)

        MAX_THREAD = self.device_attrs.MAX_THREADS_PER_BLOCK
        thread_count =  min(MAX_THREAD, N)
        block_count = int(math.ceil(float(N) / MAX_THREAD))

        # TODO: Dispatch from actual ufunc
        ufunc = np.add
        assert all(isinstance(array, np.ndarray) for array in broadcast_arrays)
        cuda.invoke_cuda_ufunc(ufunc, &self.device_attrs, cuda_func.cu_function,
                               broadcast_arrays, out, False, True,
                               block_count, 1, 1,
                               MAX_THREAD, 1, 1)

    def __dealloc__(self):
        cuda.dealloc(self.cu_module, self.cu_context)

