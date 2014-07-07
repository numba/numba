from __future__ import absolute_import, division, print_function
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numbapro.common.deviceufunc import UFuncMechanism


class CudaUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels
        self._maxblocksize = 0   # ignored

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
        return CudaUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        assert len(list(self.functions.keys())[0]) == 2, "must be a binary " \
                                                         "ufunc"
        assert arg.ndim == 1, "must use 1d array"

        n = arg.shape[0]
        gpu_mems = []

        if n == 0:
            raise TypeError("Reduction on an empty array.")
        elif n == 1:    # nothing to do
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
        else: # even?
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


class CUDAGenerializedUFunc(object):
    def __init__(self, kernelmap, engine):
        self.kernelmap = kernelmap
        self.engine = engine
        self.max_blocksize = 2 ** 30
        assert self.engine.nout == 1, "only support single output"

    def __call__(self, *args, **kws):

        is_device_array = [devicearray.is_cuda_ndarray(a) for a in args]
        if any(is_device_array) != all(is_device_array):
            raise TypeError('if device array is used, '
                            'all arguments must be device array.')
        out = kws.get('out')
        stream = kws.get('stream', 0)

        need_cuda_conv = not any(is_device_array)
        if need_cuda_conv:
            inputs = [np.array(a) for a in args]
        else:
            inputs = args

        input_shapes = [a.shape for a in inputs]
        schedule = self.engine.schedule(input_shapes)

        # find kernel
        idtypes = tuple(i.dtype for i in inputs)
        outdtype, kernel = self.kernelmap[idtypes]

        # check output
        if out is not None and schedule.output_shapes[0] != out.shape:
            raise ValueError('output shape mismatch')

        # prepare inputs
        if need_cuda_conv:
            params = [cuda.to_device(a, stream=stream)
                      for a in inputs]
        else:
            params = list(inputs)

        # allocate output
        if need_cuda_conv or out is None:
            retval = cuda.device_array(shape=schedule.output_shapes[0],
                                       dtype=outdtype, stream=stream)
        else:
            retval = out

        # execute
        assert schedule.loopn > 0, "zero looping dimension"

        if not schedule.loopdims:
            newparams = [p.reshape(1, *p.shape) for p in params]
            newretval = retval.reshape(1, *retval.shape)
            self._launch_kernel(kernel, schedule.loopn, stream,
                                newparams + [newretval])

        elif len(schedule.loopdims) > 1:
            odim = schedule.loopn
            newparams = [p.reshape(odim, *cs) for p, cs in
                         zip(params, schedule.ishapes)]
            newretval = retval.reshape(odim, *schedule.oshapes[0])
            self._launch_kernel(kernel, schedule.loopn, stream,
                                newparams + [newretval])

        else:
            self._launch_kernel(kernel, schedule.loopn, stream,
                                params + [retval])

        # post execution
        if need_cuda_conv:
            out = retval.copy_to_host(out, stream=stream)
        elif out is None:
            out = retval
        return out

    def _launch_kernel(self, kernel, nelem, stream, args):
        max_threads = min(cuda.get_current_device().MAX_THREADS_PER_BLOCK,
                          self.max_blocksize)
        ntid = self._apply_autotuning(kernel, max_threads)
        ncta = (nelem + ntid - 1) // ntid
        kernel[ncta, ntid, stream](*args)

    def _apply_autotuning(self, func, max_threads):
        # TODO
        return max_threads
        # try:
        #     atune = func.autotune
        # except RuntimeError:
        #     return max_threads
        # else:
        #     max_threads = atune.best()
        #
        #     if not max_threads:
        #         raise Exception("insufficient resources to run kernel "
        #                         "at any thread-per-block.")
        #
        #     return max_threads


class CudaUFuncMechanism(UFuncMechanism):
    """
    Provide OpenCL specialization
    """
    DEFAULT_STREAM = 0
    ARRAY_ORDER = 'A'

    def is_device_array(self, obj):
        return devicearray.is_cuda_ndarray(obj)

    def to_device(self, hostary, stream):
        return cuda.to_device(hostary, stream=stream)

    def launch(self, func, count, stream, args):
        func.forall(count, stream=stream)(*args)

    def device_array(self, shape, dtype, stream):
        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)
