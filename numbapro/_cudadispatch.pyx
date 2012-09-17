cimport dispatch
cimport cuda
cimport numpy as cnp

import os
import math
import logging

import numpy as np

include "miniutils.pyx"

cuda.init_cuda_exc_type()
cnp.import_array()

cdef int get_device_number(int device_number):
    "Get the CUDA device number from ~/.cuda_device, or -1."
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
    """
    Get the cuda device from a device number. Provide -1 to get the
    first working device.
    """
    cdef cuda.CUdevice result

    device_number = get_device_number(device_number)
    cuda.get_device(&result, NULL, device_number)
    return result

def compute_capability(int device_number):
    "Get the CUDA compute capability of the device"
    cdef cuda.CudaDeviceAttrs device_attrs
    cdef cuda.CUdevice cu_device = get_device(device_number)

    cuda.init_attributes(cu_device, &device_attrs)
    return (device_attrs.COMPUTE_CAPABILITY_MAJOR,
            device_attrs.COMPUTE_CAPABILITY_MINOR)

cdef class CudaFunction(dispatch.Function):
    "Wrap a compiled CUDA function"
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
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

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

        # print ptx_code

        self.functions = {}
        for dtypes, (result_dtype, name) in types_to_name.items():
            func = CudaFunction(name)
            func.load(&self.cu_module)
            self.functions[dtypes] = (result_dtype, func)

    def broadcast_inputs(self, args):
        # prepare broadcasted contiguous arrays
        # TODO: Allow strided memory (use mapped memory + strides?)
        # TODO: don't perform actual broadcasting, pass in strides
        args = [np.ascontiguousarray(a) for a in args]
        broadcast_arrays = np.broadcast_arrays(*args)
        return broadcast_arrays

    def allocate_output(self, broadcast_arrays, result_dtype):
        return np.empty_like(broadcast_arrays[0], dtype=result_dtype)

    def __call__(self, cnp.ufunc ufunc, *args):
        cdef CudaFunction cuda_func
        cdef cnp.npy_intp N, MAX_THREAD, thread_count, block_count

        dtypes = tuple(a.dtype for a in args)
        if dtypes not in self.functions:
            raise TypeError("Input dtypes not supported by ufunc %s" % (dtypes,))

        result_dtype, cuda_func = self.functions[dtypes]

        broadcast_arrays = self.broadcast_inputs(args)
        N = np.prod(broadcast_arrays[0].shape)

        out = self.allocate_output(broadcast_arrays, result_dtype)

        MAX_THREAD = self.device_attrs.MAX_THREADS_PER_BLOCK
        thread_count =  min(MAX_THREAD, N)
        block_count = int(math.ceil(float(N) / MAX_THREAD))

        # TODO: Dispatch from actual ufunc
        assert all(isinstance(array, np.ndarray) for array in broadcast_arrays)
        cuda.invoke_cuda_ufunc(ufunc, &self.device_attrs, cuda_func.cu_function,
                               broadcast_arrays, out, False, True,
                               block_count, 1, 1,
                               MAX_THREAD, 1, 1)
        return out

    def __dealloc__(self):
        cuda.dealloc(self.cu_module, self.cu_context)


cdef class CudaGeneralizedUFuncDispatcher(CudaUFuncDispatcher):
    """
    Implements a generalized CUDA function.
    """

    def __call__(self, cnp.ufunc ufunc, *args):
        cdef int i
        cdef cnp.npy_intp *shape_p, *strides_p
        cdef cnp.npy_intp **strides_args
        cdef char **data_pointers

        assert ufunc.nin == len(args)

        # number of core dimensions per input
        core_dimensions = []
        for i, array in enumerate(args):
            core_dimensions.append(array.ndim - ufunc.core_num_dims[i])

        arrays = [np.asarray(a) for a in args]
        broadcast = np.broadcast(*arrays)
        ndim = broadcast.nd
        broadcast_arrays(arrays, broadcast.shape, ndim, &shape_p, &strides_p)
        build_dynamic_args(arrays, strides_p, &data_pointers, &strides_args,
                           ndim)

        # TODO: finish