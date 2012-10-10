import numbapro
cimport dispatch
cimport cuda
cimport numpy as cnp

import sys
import os
import math
import logging
import ctypes

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

cdef void* addressof(x):
    cdef unsigned long long addr
    addr = ctypes.cast(x, ctypes.c_void_p).value
    return <void *>addr

cdef int _init_cuda_api() except *:
    cdef cuda.CudaAPI * cuda_api

    if not cuda.is_cuda_api_initialized():

        # Determine platform and path of cuda driver
        if sys.platform == 'win32':
            dlloader = ctypes.WinDLL
            path = '\\windows\\system32\\nvcuda.dll'
        else:
            dlloader = ctypes.CDLL
            path = '/usr/lib/libcuda.so'

        # Environment variable always overide if present
        path = os.environ.get('NUMBAPRO_CUDA_DRIVER', path)

        # Load the driver
        try:
            driver = dlloader(path)
        except OSError:
            raise ImportError(
                      "CUDA is not supported or the library cannot be found. "
                      "Try setting environment variable NUMBAPRO_CUDA_DRIVER "
                      "with the path of the CUDA driver shared library.")


        # Begin to populate the CUDA API function table
        api = cuda.get_cuda_api_ref()

        api.Init = addressof(driver.cuInit)

        api.DeviceGetCount = addressof(driver.cuDeviceGetCount)
        api.DeviceGet = addressof(driver.cuDeviceGet)
        api.DeviceGetAttribute = addressof(driver.cuDeviceGetAttribute)
        api.DeviceComputeCapability = addressof(driver.cuDeviceComputeCapability)

        api.ModuleLoadDataEx = addressof(driver.cuModuleLoadDataEx)
        api.ModuleUnload = addressof(driver.cuModuleUnload)
        api.ModuleGetFunction = addressof(driver.cuModuleGetFunction)

        api.StreamCreate = addressof(driver.cuStreamCreate)
        api.StreamSynchronize = addressof(driver.cuStreamSynchronize)

        api.LaunchKernel = addressof(driver.cuLaunchKernel)

        try:
            api.CtxCreate = addressof(driver.cuCtxCreate_v2)
            api.MemAlloc = addressof(driver.cuMemAlloc_v2)
            api.MemcpyHtoD = addressof(driver.cuMemcpyHtoD_v2)
            api.MemcpyHtoDAsync = addressof(driver.cuMemcpyHtoDAsync_v2)
            api.MemcpyDtoH = addressof(driver.cuMemcpyDtoH_v2)
            api.MemcpyDtoHAsync = addressof(driver.cuMemcpyDtoHAsync_v2)
            api.MemFree = addressof(driver.cuMemFree_v2)
            api.StreamDestroy = addressof(driver.cuStreamDestroy_v2)
        except AttributeError:
            api.CtxCreate = addressof(driver.cuCtxCreate)
            api.MemAlloc = addressof(driver.cuMemAlloc)
            api.MemcpyHtoD = addressof(driver.cuMemcpyHtoD)
            api.MemcpyHtoDAsync = addressof(driver.cuMemcpyHtoDAsync)
            api.MemcpyDtoH = addressof(driver.cuMemcpyDtoH)
            api.MemcpyDtoHAsync = addressof(driver.cuMemcpyDtoHAsync)
            api.MemFree = addressof(driver.cuMemFree)
            api.StreamDestroy = addressof(driver.cuStreamDestroy)

        cuda.set_cuda_api_initialized()

cdef cuda.CUdevice get_device(int device_number) except *:
    """
    Get the cuda device from a device number. Provide -1 to get the
    first working device.
    """
    cdef cuda.CUdevice result
    device_number = get_device_number(device_number)
    cuda.get_device(&result, NULL, device_number)
    return result

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

    cdef public dict functions
    cdef public dict name_to_func

    cdef cuda.CudaFunctionAndData *info

    def __init__(self, ptx_code, types_to_name, device_number):
        cdef CudaFunction func

        cuda.get_device(&self.cu_device, &self.cu_context, device_number)
        cuda.init_attributes(self.cu_device, &self.device_attrs)
        cuda.cuda_load(ptx_code, &self.cu_module)

        # print ptx_code

        self.functions = {}
        self.name_to_func = {}
        for dtypes, (result_dtype, name) in types_to_name.items():
            func = CudaFunction(name)
            func.load(&self.cu_module)
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
        return np.empty(broadcast_arrays[0].shape, dtype=result_dtype)

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

    def build_datalist(self, list func_names):
        """
        This is called for generalized CUDA ufuncs. They do not go through
        __call__, but rather call the functions directly. Our function needs
        some extra info which we build here.
        """
        cdef CudaFunction func
        cdef int i, nops = len(self.functions.keys()[0])

        self.info = <cuda.CudaFunctionAndData *> stdlib.malloc(
                        len(func_names) * sizeof(cuda.CudaFunctionAndData))
        if self.info == NULL:
            raise MemoryError

        result = []
        for i, func_name in enumerate(func_names):
            func = self.name_to_func[func_name]
            self.info[i].cu_func = func.cu_function
            self.info[i].nops = nops
            # numbapro.drop_in_gdb(addr=<dispatch.Py_uintptr_t> &self.info[i].nops)
            result.append(<dispatch.Py_uintptr_t> &self.info[i])

        return result

    def __dealloc__(self):
        cuda.dealloc(self.cu_module, self.cu_context)
        stdlib.free(self.info)


def get_cuda_outer_loop_addr():
    return <dispatch.Py_uintptr_t> &cuda.cuda_outer_loop

cdef class CudaGeneralizedUFuncDispatcher(CudaUFuncDispatcher):
    """
    Implements a generalized CUDA function.
    """

    def __call__(self, cnp.ufunc ufunc, *args):
        cdef int i
        cdef cnp.npy_intp *outer_shape_p, *outer_strides_p
        cdef cnp.npy_intp *inner_shape_p, *inner_strides_p

        cdef cnp.npy_intp **strides_args
        cdef char **data_pointers
        cdef int core_dims
        cdef int ndim = 0, core_ndim = 0

        assert ufunc.nin + ufunc.nout == len(args)

        args = [np.asarray(arg) for arg in args]

        # number of core dimensions per input
        core_dimensions = []
        for i, array in enumerate(args):
            core_dims = ufunc.core_num_dims[i]
            ndim = max(ndim, cnp.PyArray_NDIM(array) - core_dims)
            core_ndim = max(core_ndim, core_dims)
            core_dimensions.append(core_dims)

        arrays = [np.asarray(a) for a in args]
#        broadcast_arrays(arrays, None, ndim, &outer_shape_p, &outer_strides_p,
#                         False)
#        broadcast_arrays(arrays, None, ndim, &inner_shape_p, &inner_strides_p,
#                         True)


#cdef class CudaNumbaFuncDispatcher(object): #cutils.UFuncDispatcher):
#    """
#    Invoke the CUDA ufunc specialization for the given inputs.
#    """

#    cdef cuda.CudaDeviceAttrs device_attrs

#    cdef cuda.CUdevice cu_device
#    cdef cuda.CUcontext cu_context
#    cdef cuda.CUmodule cu_module
#    cdef cuda.CUfunction cu_function

#    cdef char* typemap

#    def __init__(self, ptx_code, func_name, device_number, typemap):
#        cuda.get_device(&self.cu_device, &self.cu_context, device_number)
#        cuda.init_attributes(self.cu_device, &self.device_attrs)
#        cuda.cuda_load(ptx_code, &self.cu_module)
#        cuda.cuda_getfunc(self.cu_module, &self.cu_function, func_name)
#        self.typemap = typemap

#    def __call__(self, args, griddim, blkdim):
#        gx, gy, gz = griddim
#        bx, by, bz = blkdim
#        cuda.cuda_numba_function(list(args), self.cu_function,
#                                 gx, gy, gz, bx, by, bz,
#                                 self.typemap)

#    def __dealloc__(self):
#        cuda.dealloc(self.cu_module, self.cu_context)

