# Raise ImportError if we cannot find CUDA Driver.

import numpy as np
from ctypes import c_float, c_double, c_int
from _cudadispatchlib import *
from numbapro._cuda import driver as _cuda
from numbapro._cuda import default as _cudadefaults
from numbapro._cuda.ndarray import ndarray_to_device_memory, \
                                   ndarray_data_to_device_memory

def compute_capability():
    "Get the CUDA compute capability of the device"
    return _cudadefaults.device.COMPUTE_CAPABILITY


class CudaUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """
    def __init__(self, ptx_code, types_to_name, device_number):
        cu_module = _cuda.Module(ptx_code)

        self.functions = {}
        self.name_to_func = {}
        for dtypes, (result_dtype, name) in types_to_name.items():
            func = _cuda.Function(cu_module, name)
            self.functions[dtypes] = (result_dtype, func)
            self.name_to_func[name] = func

    def broadcast_inputs(self, args):
        # prepare broadcasted contiguous arrays
        # TODO: Allow strided memory (use mapped memory + strides?)
        # TODO: don't perform actual broadcasting, pass in strides
        args = [np.ascontiguousarray(a) for a in args]
        broadcast_arrays = np.broadcast_arrays(*args)
        return broadcast_arrays

    def allocate_output(self, broadcast_arrays, result_dtype):
        # return np.empty_like(broadcast_arrays[0], dtype=result_dtype)
        # for numpy1.5
        return np.zeros(broadcast_arrays[0].shape, dtype=result_dtype)

    def __call__(self, ufunc, *args):
        dtypes = tuple(a.dtype for a in args)
        if dtypes not in self.functions:
            raise TypeError("Input dtypes not supported by ufunc %s" % (dtypes,))

        result_dtype, cuda_func = self.functions[dtypes]

        broadcast_arrays = self.broadcast_inputs(args)
        element_count = np.prod(broadcast_arrays[0].shape)

        out = self.allocate_output(broadcast_arrays, result_dtype)

        MAX_THREAD = cuda_func.device.MAX_THREADS_PER_BLOCK
        thread_count =  min(MAX_THREAD, element_count)
        block_count = int(math.ceil(float(element_count) / MAX_THREAD))

        # TODO: Dispatch from actual ufunc
        assert all(isinstance(array, np.ndarray) for array in broadcast_arrays)

        with _cuda.Stream() as stream:
            kernel_args = [dvmem for _, dvmem
                                 in map(ndarray_data_to_device_memory,
                                        broadcast_arrays)]
            retriever, output_args = ndarray_data_to_device_memory(out)
            kernel_args.append(output_args)
            kernel_args.append(c_int(element_count))

            griddim = (block_count,)
            blockdim = (thread_count,)

            cu_func = cuda_func.configure(griddim, blockdim, stream=stream)
            cu_func(*kernel_args)

            retriever() # only retrive the last one
        return out

    def build_datalist(self, func_names):
        """
        This is called for generalized CUDA ufuncs. They do not go through
        __call__, but rather call the functions directly. Our function needs
        some extra info which we build here.
        """
        raise Exception("Need to verify this.")
        #        i, nops = len(self.functions.keys()[0])

        #        self.info = <cuda.CudaFunctionAndData *> stdlib.malloc(
        #                        len(func_names) * sizeof(cuda.CudaFunctionAndData))
        #        if self.info == NULL:
        #            raise MemoryError

        #        result = []
        #        for i, func_name in enumerate(func_names):
        #            func = self.name_to_func[func_name]
        #            self.info[i].cu_func = func.cu_function
        #            self.info[i].nops = nops
        #            # numbapro.drop_in_gdb(addr=<dispatch.Py_uintptr_t> &self.info[i].nops)
        #            result.append(<dispatch.Py_uintptr_t> &self.info[i])

        #        return result



class CudaNumbaFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, ptx_code, func_name, device_number, typemap):
        cu_module = _cuda.Module(ptx_code)
        self.cu_function = _cuda.Function(cu_module, func_name)
        self.typemap = typemap

    def __call__(self, args, griddim, blkdim):
        kernel_args = []

        with _cuda.Stream() as stream:
            retrievers = []
            def ndarray_gpu(x):
                retriever, device_memory = ndarray_to_device_memory(x, stream=stream)
                retrievers.append(retriever)
                return device_memory

            _typemapper = {'f': c_float,
                           'd': c_double,
                           'i': c_int,
                           '_': ndarray_gpu}


            for ty, arg in zip(self.typemap, args):
                kernel_args.append(_typemapper[ty](arg))

            cu_func = self.cu_function.configure(griddim, blkdim, stream=stream)
            cu_func(*kernel_args)

            for r in retrievers:
                r()


