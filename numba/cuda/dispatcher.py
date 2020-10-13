import numpy as np

from numba import cuda
from numba.cuda.cudadrv import devicearray
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GenerializedUFunc,
                                        GUFuncCallSteps)


class CUDAUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels
        self._maxblocksize = 0  # ignored

    @property
    def max_blocksize(self):
        return self._maxblocksize

    @max_blocksize.setter
    def max_blocksize(self, blksz):
        self._max_blocksize = blksz

    def __call__(self, *args, **kws):
        """
        *args: numpy arrays or DeviceArrayBase (created by cuda.to_device).
               Cannot mix the two types in one call.

        **kws:
            stream -- cuda stream; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return CUDAUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        assert len(list(self.functions.keys())[0]) == 2, "must be a binary " \
                                                         "ufunc"
        assert arg.ndim == 1, "must use 1d array"

        n = arg.shape[0]
        gpu_mems = []

        if n == 0:
            raise TypeError("Reduction on an empty array.")
        elif n == 1:  # nothing to do
            return arg[0]

        # always use a stream
        stream = stream or cuda.stream()
        with stream.auto_synchronize():
            # transfer memory to device if necessary
            if devicearray.is_cuda_ndarray(arg):
                mem = arg
            else:
                mem = cuda.to_device(arg, stream)
                # do reduction
            out = self.__reduce(mem, gpu_mems, stream)
            # use a small buffer to store the result element
            buf = np.array((1,), dtype=arg.dtype)
            out.copy_to_host(buf, stream=stream)

        return buf[0]

    def __reduce(self, mem, gpu_mems, stream):
        n = mem.shape[0]
        if n % 2 != 0:  # odd?
            fatcut, thincut = mem.split(n - 1)
            # prevent freeing during async mode
            gpu_mems.append(fatcut)
            gpu_mems.append(thincut)
            # execute the kernel
            out = self.__reduce(fatcut, gpu_mems, stream)
            gpu_mems.append(out)
            return self(out, thincut, out=out, stream=stream)
        else:  # even?
            left, right = mem.split(n // 2)
            # prevent freeing during async mode
            gpu_mems.append(left)
            gpu_mems.append(right)
            # execute the kernel
            self(left, right, out=left, stream=stream)
            if n // 2 > 1:
                return self.__reduce(left, gpu_mems, stream)
            else:
                return left


class _CUDAGUFuncCallSteps(GUFuncCallSteps):
    __slots__ = [
        '_stream',
    ]

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary):
        return cuda.to_device(hostary, stream=self._stream)

    def to_host(self, devary, hostary):
        out = devary.copy_to_host(hostary, stream=self._stream)
        return out

    def device_array(self, shape, dtype):
        return cuda.device_array(shape=shape, dtype=dtype, stream=self._stream)

    def prepare_inputs(self):
        self._stream = self.kwargs.get('stream', 0)

    def launch_kernel(self, kernel, nelem, args):
        kernel.forall(nelem, stream=self._stream)(*args)


class CUDAGenerializedUFunc(GenerializedUFunc):
    @property
    def _call_steps(self):
        return _CUDAGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        return devicearray.DeviceNDArray(shape=shape,
                                         strides=(0,),
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)

    def _broadcast_add_axis(self, ary, newshape):
        newax = len(newshape) - len(ary.shape)
        # Add 0 strides for missing dimension
        newstrides = (0,) * newax + ary.strides
        return devicearray.DeviceNDArray(shape=newshape,
                                         strides=newstrides,
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)


class CUDAUFuncMechanism(UFuncMechanism):
    """
    Provide OpenCL specialization
    """
    DEFAULT_STREAM = 0

    def launch(self, func, count, stream, args):
        func.forall(count, stream=stream)(*args)

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary, stream):
        return cuda.to_device(hostary, stream=stream)

    def to_host(self, devary, stream):
        return devary.copy_to_host(stream=stream)

    def device_array(self, shape, dtype, stream):
        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)

    def broadcast_device(self, ary, shape):
        ax_differs = [ax for ax in range(len(shape))
                      if ax >= ary.ndim
                      or ary.shape[ax] != shape[ax]]

        missingdim = len(shape) - len(ary.shape)
        strides = [0] * missingdim + list(ary.strides)

        for ax in ax_differs:
            strides[ax] = 0

        return devicearray.DeviceNDArray(shape=shape,
                                         strides=strides,
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)
