from __future__ import absolute_import, division, print_function

import numpy as np

from numba.npyufunc.deviceufunc import (UFuncMechanism, GenerializedUFunc,
                                        GUFuncCallSteps)


class HsaUFuncDispatcher(object):
    """
    Invoke the HSA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels

    def __call__(self, *args, **kws):
        """
        *args: numpy arrays
        **kws:
            stream -- hsa stream; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return HsaUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        raise NotImplementedError


class HsaUFuncMechanism(UFuncMechanism):
    """
    Provide OpenCL specialization
    """
    DEFAULT_STREAM = 0
    ARRAY_ORDER = 'A'

    def is_device_array(self, obj):
        return isinstance(obj, np.ndarray)

    def is_host_array(self, obj):
        return isinstance(obj, np.ndarray)

    def to_device(self, hostary, stream):
        return hostary

    def launch(self, func, count, stream, args):
        # ILP must match vectorize kernel source
        ilp = 4
        # Use more wavefront to allow hiding latency
        tpb = 64 * 2
        count = (count + (ilp - 1)) // ilp
        blockcount = (count + (tpb - 1)) // tpb
        func[blockcount, tpb](*args)

    def device_array(self, shape, dtype, stream):
        return np.empty(shape=shape, dtype=dtype)

    def broadcast_device(self, ary, shape):
        ax_differs = [ax for ax in range(len(shape))
                      if ax >= ary.ndim
                      or ary.shape[ax] != shape[ax]]

        missingdim = len(shape) - len(ary.shape)
        strides = [0] * missingdim + list(ary.strides)

        for ax in ax_differs:
            strides[ax] = 0

        return np.ndarray(shape=shape, strides=strides,
                          dtype=ary.dtype, buffer=ary)


class _HsaGUFuncCallSteps(GUFuncCallSteps):
    __slots__ = ()

    def is_device_array(self, obj):
        return True

    def to_device(self, hostary):
        return hostary

    def to_host(self, devary, hostary):
        pass

    def device_array(self, shape, dtype):
        return np.empty(shape=shape, dtype=dtype)

    def launch_kernel(self, kernel, nelem, args):
        kernel.configure(nelem, min(nelem, 64))(*args)


class HSAGenerializedUFunc(GenerializedUFunc):
    @property
    def _call_steps(self):
        return _HsaGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        return np.lib.stride_tricks.as_strided(ary, shape=(shape,),
                                               strides=(0,))
