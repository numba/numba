# Raise ImportError if we cannot find CUDA Driver.

from llvm import core as _lc
import numpy as np
from ctypes import *
from numbapro._cuda import driver as _cuda
from numbapro._cuda import default as _cudadefaults
from numbapro._cuda.ndarray import ndarray_to_device_memory,      \
                                   ndarray_data_to_device_memory, \
                                   NumpyStructure
from numbapro._cuda.devicearray import DeviceNDArray
from numbapro import cuda
import math

def compute_capability():
    "Get the CUDA compute capability of the device"
    return _cudadefaults.device.COMPUTE_CAPABILITY

class CudaUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """
    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels

    def broadcast_inputs(self, args):
        # prepare broadcasted contiguous arrays
        # TODO: Allow strided memory (use mapped memory + strides?)
        # TODO: don't perform actual broadcasting, pass in strides
        args = [np.ascontiguousarray(a) for a in args]
        first_ary = args[0]
        broadcast_arrays = np.broadcast_arrays(*args)
        return broadcast_arrays

    def allocate_output(self, broadcast_arrays, result_dtype):
        # return np.empty_like(broadcast_arrays[0], dtype=result_dtype)
        # for numpy1.5
        return np.empty(broadcast_arrays[0].shape, dtype=result_dtype)

    def __call__(self, *args, **kws):
        assert not kws or 'stream' in kws
        stream = kws.get('stream', 0)
        dtypes = tuple(a.dtype for a in args)
        if dtypes not in self.functions:
            raise TypeError("Input dtypes not supported by ufunc %s" % (dtypes,))

        # find the fitting function
        result_dtype, cuda_func = self.functions[dtypes]
        MAX_THREAD = cuda_func.device.MAX_THREADS_PER_BLOCK

        is_device_ndarray = [isinstance(x, DeviceNDArray) for x in args]

        if all(is_device_ndarray):
            # NOTE: When using DeviceNDArray,
            #       it is assumed to be properly broadcasted.
            assert args[0].ndim == 1
            assert all(x.shape == args.shape for x in args)

            element_count = args[0].shape[0]
            out = np.empty(args[0].shape[0], dtype=result_dtype)
            nctaid, ntid = self.determine_dimensions(element_count, MAX_THREAD)
            
            griddim = (nctaid,)
            blockdim = (ntid,)

            device_out = cuda.to_device(out, stream)

            kernel_args = device_ins + [device_out, element_count]

            cuda_func[griddim, blockdim, stream](*kernel_args)

            device_out.to_host(stream) # only retrive the last one

        elif not any(is_device_ndarray):
            broadcast_arrays = self.broadcast_inputs(args)
            element_count = np.prod(broadcast_arrays[0].shape)

            out = self.allocate_output(broadcast_arrays, result_dtype)
            nctaid, ntid = self.determine_dimensions(element_count, MAX_THREAD)

            assert all(isinstance(array, np.ndarray) for array in broadcast_arrays)

            device_ins = [cuda.to_device(x, stream) for x in broadcast_arrays]
            device_out = cuda.to_device(out, stream)

            kernel_args = device_ins + [device_out, element_count]

            griddim = (nctaid,)
            blockdim = (ntid,)
            
            cuda_func[griddim, blockdim, stream](*kernel_args)
            
            device_out.to_host(stream) # only retrive the last one

        else:
            raise ValueError("Cannot mix DeviceNDArray and ndarray as args")

        return device_out

    def determine_dimensions(self, n, max_thread):
        # determine grid and block dimension
        thread_count =  min(max_thread, n)
        block_count = int(math.ceil(float(n) / max_thread))
        return block_count, thread_count

class CudaFunctionAndData(Structure):
    _fields_ = [
        ('cu_func', _cuda.cu_function),
        ('nops',    c_uint),
    ]

def _cuda_outer_loop(args, dimensions, steps, data, arrays):
    info = cast(data, POINTER(CudaFunctionAndData))
    cu_func = info[0].cu_func
    nops = info[0].nops

    arrays_ptr = cast(arrays, POINTER(POINTER(NumpyStructure)))
    ndarrays = map(lambda x: x.contents,  (arrays_ptr[i] for i in range(nops)))

    driver = _cudadefaults.driver
    stream = _cuda.Stream()
    with stream.auto_synchronize():

        device_count = c_size_t(dimensions[0])
        MAX_THREAD = 128
        block_per_grid = 1
        thread_per_block = dimensions[0]

        def alloc_and_copy(data, size):
            memory = _cuda.DeviceMemory(size)
            memory.to_device_raw(data, size, stream=stream)
            return memory

        # arguments
        ref_args = []
        temp_args = (_cuda.cu_device_ptr * nops)()

        for i in range(nops):
            array = ndarrays[i]
            ptr = cast(array.data, POINTER(c_float))
            memory = alloc_and_copy(array.data, steps[i] * dimensions[0])
            ref_args.append(memory)
            temp_args[i] = memory._handle

        device_args = alloc_and_copy(addressof(temp_args), sizeof(temp_args))
        device_args.add_dependencies(*ref_args)

        #dimensions
        ref_dims = []
        temp_dims = (_cuda.cu_device_ptr * nops)()
        for i in range(nops):
            array = ndarrays[i]
            memory = alloc_and_copy(array.dimensions,
                                    sizeof(np.ctypeslib.c_intp) * array.nd)
            ref_dims.append(memory)
            temp_dims[i] = memory._handle
        device_dims = alloc_and_copy(addressof(temp_dims), sizeof(temp_dims))
        device_dims.add_dependencies(*ref_dims)

        # steps
        ref_steps = []
        temp_steps = (_cuda.cu_device_ptr * nops)()
        for i in range(nops):
            array = ndarrays[i]
            memory = alloc_and_copy(array.strides,
                                    sizeof(np.ctypeslib.c_intp) * array.nd)
            ref_steps.append(memory)
            temp_steps[i] = memory._handle
        device_steps = alloc_and_copy(addressof(temp_steps), sizeof(temp_steps))
        device_steps.add_dependencies(*ref_steps)

        # outer loop step
        device_arylen = alloc_and_copy(steps, sizeof(np.ctypeslib.c_intp) * nops)

        # launch
        if thread_per_block >= MAX_THREAD:
            block_per_grid = int(math.ceil(float(thread_per_block) / MAX_THREAD))
            thread_per_block = MAX_THREAD

        griddim = block_per_grid, 1, 1
        blockdim = thread_per_block, 1, 1

        kernel_args = [device_args, device_dims, device_steps, device_arylen,
                       device_count]

        _cuda.launch_kernel(cu_func, griddim, blockdim, 0, stream._handle, kernel_args)

        # retrieve
        ref_args[-1].from_device_raw(ndarrays[nops - 1].data,
                                     steps[nops - 1] * dimensions[0])


cuda_outer_loop = CFUNCTYPE(None,
                            POINTER(c_char_p),
                            POINTER(np.ctypeslib.c_intp),
                            POINTER(np.ctypeslib.c_intp),
                            c_void_p,
                            POINTER(py_object),)(_cuda_outer_loop)

def get_cuda_outer_loop_addr():
    return cast(cuda_outer_loop, c_void_p).value


class CudaGeneralizedUFuncDispatcher(CudaUFuncDispatcher):
    """
    Implements a generalized CUDA function.
    """

    def __call__(self, ufunc, *args):
        raise
        ndim = 0
        core_ndim = 0

        assert ufunc.nin + ufunc.nout == len(args)

        args = [np.asarray(arg) for arg in args]

        # number of core dimensions per input
        core_dimensions = []
        for i, array in enumerate(args):
            core_dims = ufunc.core_num_dims[i]
            ndim = max(ndim, len(array.shape) - core_dims)
            core_ndim = max(core_ndim, core_dims)
            core_dimensions.append(core_dims)

        arrays = [np.asarray(a) for a in args]

def _apply_typemap(lfunctype):
    argtys = lfunctype.pointee.args
    def convert(ty):
        if isinstance(ty, _lc.IntegerType):
            return 'i'
        elif ty == _lc.Type.float():
            return 'f'
        elif ty == _lc.Type.double():
            return 'd'
        else:
            return '_'
    return ''.join(map(convert, argtys))

class CudaNumbaFuncDispatcher(object):

    def __init__(self, ptx_code, func_name, lfunctype):
        cu_module = _cuda.Module(ptx_code)
        self.cu_function = _cuda.Function(cu_module, func_name)
        self.typemap = _apply_typemap(lfunctype)
        # default to prefer cache
        # self.cu_function.cache_config(prefer_shared=True)

    @property
    def device(self):
        return self.cu_function.device

    def _cast_args(self, args):
        # Cast scalar arguments to match the prototype.
        def convert(ty, val):
            if ty == 'f' or ty == 'd':
                return float(val)
            elif ty == 'i':
                return int(val)
            else:
                return val
        return [convert(ty, val) for ty, val in zip(self.typemap, args)]

    def __call__(self, args, griddim, blkdim, stream=0):
        from ._cuda.devicearray import DeviceNDArray
        args = self._cast_args(args)

        kernel_args = []

        retrievers = []
        def ndarray_gpu(x):
            if isinstance(x, DeviceNDArray):
                return x.device_memory
            else:
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

