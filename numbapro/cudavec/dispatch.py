from __future__ import absolute_import, division, print_function
import math
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba.cuda import get_current_device
from numba import cuda


class CudaUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels

    @property
    def max_blocksize(self):
        try:
            return self.__max_blocksize
        except AttributeError:
            return 2 ** 30 # a very large number

    @max_blocksize.setter
    def max_blocksize(self, blksz):
        self.__max_blocksize = blksz

    @max_blocksize.deleter
    def max_blocksize(self, blksz):
        del self.__max_blocksize

    def _prepare_inputs(self, args):
        # prepare broadcasted contiguous arrays
        # TODO: Allow strided memory (use mapped memory + strides?)
        # TODO: don't perform actual broadcasting, pass in strides
        #        args = [np.ascontiguousarray(a) for a in args]

        return np.broadcast_arrays(*args)

    def _adjust_dimension(self, broadcast_arrays):
        '''Reshape the broadcasted arrays so that they are all 1D arrays.
        Uses ndarray.ravel() to flatten.  It only copy if necessary.
        '''
        for i, ary in enumerate(broadcast_arrays):
            if ary.ndim > 1: # flatten multi-dimension arrays
                broadcast_arrays[i] = ary.ravel() # copy if necessary
        return broadcast_arrays

    def _allocate_output(self, broadcast_arrays, result_dtype):
        return np.empty(shape=broadcast_arrays[0].shape, dtype=result_dtype)

    def _apply_autotuning(self, func, max_threads):
        try:
            atune = func.autotune
        except RuntimeError:
            return max_threads
        else:
            max_threads = atune.best()

            if not max_threads:
                raise Exception("insufficient resources to run kernel"
                                "at any thread-per-block.")

            return max_threads

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
        accepted_kws = 'stream', 'out'
        unknown_kws = [k for k in kws if k not in accepted_kws]
        assert not unknown_kws, ("Unknown keyword args %s" % unknown_kws)

        stream = kws.get('stream', 0)

        # convert arguments to ndarray if they are not
        args = list(args) # convert to list
        has_device_array_arg = any(devicearray.is_cuda_ndarray(v)
                                   for v in args)

        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray) and \
                    not devicearray.is_cuda_ndarray(arg):
                args[i] = ary = np.asarray(arg)

        # get the dtype for each argument
        def _get_dtype(x):
            try:
                return x.dtype
            except AttributeError:
                return np.dtype(type(x))

        dtypes = tuple(_get_dtype(a) for a in args)

        # find the fitting function
        result_dtype, cuda_func = self._get_function_by_dtype(dtypes)

        device_maxthreads = get_current_device().MAX_THREADS_PER_BLOCK
        max_threads = min(device_maxthreads, self.max_blocksize)

        # apply autotune
        if max_threads == device_maxthreads:
            # TODO
            pass
            #max_threads = self._apply_autotuning(cuda_func, max_threads)

        if has_device_array_arg:
            # Ugly: convert array scalar into zero-strided one element array.
            for i, ary in enumerate(args):
                if not ary.shape:
                    data = np.asscalar(ary)
                    ary = np.ndarray(shape=(1,), strides=(0,))
                    ary[0] = data
                    args[i] = ary

            # NOTE: When using DeviceArrayBase,
            #       it is assumed to be properly broadcasted.

            self._arguments_requirement(args)
            args, argconv = zip(*(cuda._auto_device(a) for a in args))

            element_count = self._determine_element_count(args)
            nctaid, ntid = self._determine_dimensions(element_count,
                                                      max_threads)

            griddim = (nctaid,)
            blockdim = (ntid,)

            if 'out' not in kws:
                out_shape = self._determine_output_shape(args)
                device_out = cuda.device_array(shape=out_shape,
                                               dtype=result_dtype,
                                               stream=stream)
            else:
                device_out = kws['out']
                if not devicearray.is_cuda_ndarray(device_out):
                    raise TypeError("output array must be a device array")

            def flatten_args(a):
                if a.ndim > 1:
                    a = a.reshape(np.prod(a.shape))
                    return a
                return a

            kernel_args = [flatten_args(a) for a in args]
            kernel_args.append(flatten_args(device_out))
            kernel_args.append(element_count)

            cuda_func[griddim, blockdim, stream](*kernel_args)

            return device_out

        else:
            broadcast_arrays = self._prepare_inputs(args)
            element_count = self._determine_element_count(broadcast_arrays)

            if 'out' not in kws:
                out = self._allocate_output(broadcast_arrays, result_dtype)
            else:
                out = kws['out']
                if devicearray.is_cuda_ndarray(out):
                    raise TypeError("output array must not be a device array")
                if out.shape[0] < broadcast_arrays[0].shape[0]:
                    raise ValueError("insufficient storage for output array")

            # Reshape the arrays if necessary.
            # Ufunc expects 1D array.
            reshape = out.shape
            (out,) = self._adjust_dimension([out])
            broadcast_arrays = self._adjust_dimension(broadcast_arrays)

            nctaid, ntid = self._determine_dimensions(element_count,
                                                      max_threads)

            assert all(isinstance(array, np.ndarray)
                       for array in broadcast_arrays), \
                "not all arrays are numpy ndarray"

            device_ins = [cuda.to_device(x, stream) for x in broadcast_arrays]
            device_out = cuda.device_array_like(out, stream=stream)

            kernel_args = device_ins + [device_out]

            griddim = (nctaid,)
            blockdim = (ntid,)

            cuda_func[griddim, blockdim, stream](*kernel_args)

            device_out.copy_to_host(out, stream) # only retrive the last one
            # Revert the shape of the array if it has been modified earlier
            return out.reshape(reshape)


    def _determine_output_shape(self, broadcast_arrays):
        return broadcast_arrays[0].shape

    def _get_function_by_dtype(self, dtypes):
        try:
            result_dtype, cuda_func = self.functions[dtypes]
            return result_dtype, cuda_func
        except KeyError:
            raise TypeError("Input dtypes not supported by ufunc %s" %
                            (dtypes,))

    def _determine_element_count(self, broadcast_arrays):
        return np.prod(broadcast_arrays[0].shape)

    def _arguments_requirement(self, args):
        # get shape of all array
        array_shapes = []
        for i, a in enumerate(args):
            if a.strides[0] != 0:
                array_shapes.append((i, a.shape[0]))

        _, ms = array_shapes[0]
        for i, s in array_shapes[1:]:
            if ms != s:
                raise ValueError("arg %d should have length %d" % ms)

    def _determine_dimensions(self, n, max_thread):
        # determine grid and block dimension
        thread_count = int(min(max_thread, n))
        block_count = int((n + max_thread - 1) // max_thread)
        return block_count, thread_count

    def reduce(self, arg, stream=0):
        assert len(self.functions.keys()[0]) == 2, "must be a binary ufunc"
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
        if n % 2 != 0: # odd?
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

@cuda.jit("void(float32[:], float32[:])")
def haha(a, b):
    b[0] = 0

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

