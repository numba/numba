# Raise ImportError if we cannot find CUDA Driver.

import numpy as np
from ctypes import c_float, c_double, c_int
from _cudadispatchlib import *
from numbapro._cuda import driver as _cuda
from numbapro._cuda import default as _cuglobals
from numbapro._cuda.ndarray import ndarray_to_device_memory

def compute_capability():
    "Get the CUDA compute capability of the device"
    return _cuglobals.device.COMPUTE_CAPABILITY

class CudaNumbaFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, ptx_code, func_name, device_number, typemap):
        cu_module = _cuda.Module(_cuglobals.context, ptx_code)
        self.cu_function = _cuda.Function(cu_module, func_name)
        self.typemap = typemap

    def __call__(self, args, griddim, blkdim):
        kernel_args = []

        with _cuda.Stream(self.cu_function.context) as stream:
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


