"""
Implements custom ufunc dispatch mechanism for non-CPU devices.
"""
from __future__ import print_function, absolute_import
import warnings
import numpy as np
from numba.utils import longint

class UFuncMechanism(object):
    """
    Prepare ufunc arguments for vectorize.
    """
    DEFAULT_STREAM = None
    SUPPORT_DEVICE_SLICING = False

    def __init__(self, typemap, args):
        """Never used directly by user. Invoke by UFuncMechanism.call().
        """
        self.typemap = typemap
        self.args = args
        nargs = len(self.args)
        self.argtypes = [None] * nargs
        self.scalarpos = []
        self.signature = None
        self.arrays = [None] * nargs

    def _fill_arrays(self):
        """
        Get all arguments in array form
        """
        for i, arg in enumerate(self.args):
            if isinstance(arg, np.ndarray):
                self.arrays[i] = arg
            elif self.is_device_array(arg):
                self.arrays[i] = arg
            elif isinstance(arg, (int, longint, float, complex)):
                # Is scalar
                self.scalarpos.append(i)
            else:
                raise TypeError("argument #%d has invalid type" % (i + 1,))

    def _fill_argtypes(self):
        """
        Get dtypes
        """
        for i, ary in enumerate(self.arrays):
            if ary is not None:
                self.argtypes[i] = ary.dtype

    def _resolve_signature(self):
        """Resolve signature.
        May have ambiguous case.
        """
        matches = []
        # Resolve scalar args exact match first
        if self.scalarpos:
            # Try resolve scalar arguments
            for formaltys in self.typemap:
                match_map = []
                for i, (formal, actual) in enumerate(zip(formaltys,
                                                         self.argtypes)):
                    if actual is None:
                        actual = np.asarray(self.args[i]).dtype

                    match_map.append(actual == formal)

                if all(match_map):
                    matches.append(formaltys)

        # No matching with exact match; try coercing the scalar arguments
        if not matches:
            matches = []
            for formaltys in self.typemap:
                all_matches = all(actual is None or formal == actual
                                  for formal, actual in
                                  zip(formaltys, self.argtypes))
                if all_matches:
                    matches.append(formaltys)

        if not matches:
            raise TypeError("No matching version")

        if len(matches) > 1:
            raise TypeError("Failed to resolve ufunc due to ambiguous "
                            "signature. Too many untyped scalars. "
                            "Use numpy dtype object to type tag.")

        # Try scalar arguments
        self.argtypes = matches[0]

    def _get_actual_args(self):
        """Return the actual arguments
        Casts scalar arguments to numpy.array.
        """
        for i in self.scalarpos:
            self.arrays[i] = np.array([self.args[i]], dtype=self.argtypes[i])

        return self.arrays

    def _broadcast(self, arys):
        """Perform numpy ufunc broadcasting
        """
        # Get the biggest shape
        tosort = [(a.ndim, a.size, i) for i, a in enumerate(arys)]
        biggest = max(tosort, key=lambda x: x[:-1])[-1]
        shape = arys[biggest].shape

        for i, ary in enumerate(arys):
            if ary.shape == shape:
                pass

            elif ary.shape[0] == 1:
                if self.is_device_array(ary):
                    arys[i] = self.broadcast_device(ary, shape)

                else:
                    missingdim = len(shape) - len(ary.shape) + 1
                    strides = (0,) * missingdim + ary.strides[1:]
                    strided = np.lib.stride_tricks.as_strided(ary,
                                                              shape=shape,
                                                              strides=strides)
                    arys[i] = self.force_array_layout(strided)

            else:
                print(arys)
                raise ValueError("arg #%d cannot be broadcasted" % (i + 1))

        return arys

    def get_arguments(self):
        """Prepare and return the arguments for the ufunc.
        Does not call to_device().
        """
        self._fill_arrays()
        self._fill_argtypes()
        self._resolve_signature()
        arys = self._get_actual_args()
        return self._broadcast(arys)

    def get_function(self):
        """Returns (result_dtype, function)
        """
        return self.typemap[self.argtypes]

    def is_device_array(self, obj):
        """Is the `obj` a device array?
        Override in subclass
        """
        return False

    def broadcast_device(self, ary, shape):
        """Handles ondevice broadcasting

        Override in subclass to add support.
        """
        raise NotImplementedError("broadcasting on device is not supported")

    def force_array_layout(self, ary):
        """Ensures array layout met device requirement.

        Override in sublcass
        """
        return ary

    @classmethod
    def call(cls, typemap, args, kws):
        """Perform the entire ufunc call mechanism.
        """
        # Handle keywords
        stream = kws.pop('stream', cls.DEFAULT_STREAM)
        out = kws.pop('out', None)

        if kws:
            warnings.warn("unrecognized keywords: %s" % ', '.join(kws))

        # Begin call resolution
        cr = cls(typemap, args)
        args = cr.get_arguments()
        resty, func = cr.get_function()

        outshape = args[0].shape
        if args[0].ndim > 1:
            if not cr.SUPPORT_DEVICE_SLICING:
                args = [a.ravel() for a in args]
                #raise NotImplementedError("Support 1D array only.")
            else:
                raise NotImplementedError

        # Prepare argument on the device
        devarys = []
        all_device = True
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
            else:
                dev_a = cr.to_device(a, stream=stream)
                devarys.append(dev_a)
                all_device = False

        # Launch
        shape = args[0].shape
        if out is None:
            # No output is provided
            devout = cr.device_array(shape, resty, stream=stream)

            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)

            if all_device:
                # If all arguments are on device,
                # Keep output on the device
                return devout.reshape(outshape)
            else:
                # Otherwise, transfer output back to host
                return devout.copy_to_host().reshape(outshape)

        elif cr.is_device_array(out):
            # If output is provided and it is a device array,
            # Return device array
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.reshape(outshape)

        else:
            # If output is provided and it is a host array,
            # Return host array
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.device_array(shape, resty, stream=stream)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.copy_to_host(out, stream=stream).reshape(outshape)

    def to_device(self, hostary, stream):
        """Implement to device transfer
        Override in subclass
        """
        raise NotImplementedError

    def device_array(self, shape, dtype, stream):
        """Implements device allocation
        Override in subclass
        """
        raise NotImplementedError

    def launch(self, func, count, stream, args):
        """Implements device function invocation
        Override in subclass
        """
        raise NotImplementedError
