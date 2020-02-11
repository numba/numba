import numpy as np

from numba.np.ufunc.deviceufunc import (UFuncMechanism, GenerializedUFunc,
                                        GUFuncCallSteps)
from numba.roc.hsadrv.driver import dgpu_present
import numba.roc.hsadrv.devicearray as devicearray
import numba.roc.api as api

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
        if dgpu_present:
            return devicearray.is_hsa_ndarray(obj)
        else:
            return isinstance(obj, np.ndarray)

    def is_host_array(self, obj):
        if dgpu_present:
            return False
        else:
            return isinstance(obj, np.ndarray)

    def to_device(self, hostary, stream):
        if dgpu_present:
            return api.to_device(hostary)
        else:
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
        if dgpu_present:
            return api.device_array(shape=shape, dtype=dtype)
        else:
            return np.empty(shape=shape, dtype=dtype)

    def broadcast_device(self, ary, shape):
        if dgpu_present:
            raise NotImplementedError('device broadcast_device NIY')
        else:
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
        if dgpu_present:
            return devicearray.is_hsa_ndarray(obj)
        else:
            return True

    def to_device(self, hostary):
        if dgpu_present:
            return api.to_device(hostary)
        else:
            return hostary

    def to_host(self, devary, hostary):
        if dgpu_present:
            out = devary.copy_to_host(hostary)
            return out
        else:
            pass

    def device_array(self, shape, dtype):
        if dgpu_present:
            return api.device_array(shape=shape, dtype=dtype)
        else:
            return np.empty(shape=shape, dtype=dtype)

    def launch_kernel(self, kernel, nelem, args):
        kernel.configure(nelem, min(nelem, 64))(*args)


class HSAGenerializedUFunc(GenerializedUFunc):
    @property
    def _call_steps(self):
        return _HsaGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        if dgpu_present:
            return devicearray.DeviceNDArray(shape=shape,
                                         strides=(0,),
                                         dtype=ary.dtype,
                                         dgpu_data=ary.dgpu_data)
        else:
            return np.lib.stride_tricks.as_strided(ary, shape=(shape,),
                                               strides=(0,))

    def _broadcast_add_axis(self, ary, newshape):
        newax = len(newshape) - len(ary.shape)
        # Add 0 strides for missing dimension
        newstrides = (0,) * newax + ary.strides
        if dgpu_present:
                return devicearray.DeviceNDArray(shape=newshape,
                                         strides=newstrides,
                                         dtype=ary.dtype,
                                         dgpu_data=ary.dgpu_data)
        else:
                raise NotImplementedError

