from __future__ import print_function, absolute_import
import warnings
import numpy as np
from numba import sigutils, ocl
from numba.utils import IS_PY3, longint
from numba.ocl.ocldrv import oclarray

if IS_PY3:
    def _exec(codestr, glbls):
        exec (codestr, glbls)
else:
    eval(compile("""
def _exec(codestr, glbls):
    exec codestr in glbls
""",
                 "<_exec>", "exec"))

vectorizer_stager_source = '''
def __vectorized_%(name)s(%(args)s, __out__):
    __tid__ = __ocl__.get_global_id(0)
    __out__[__tid__] = __core__(%(argitems)s)
'''


def to_dtype(ty):
    return np.dtype(str(ty))


class OclVectorize(object):
    def __init__(self, func, targetoptions={}):
        assert not targetoptions
        self.pyfunc = func
        self.kernelmap = {}  # { arg_dtype: (return_dtype), kernel }

    def add(self, sig=None, argtypes=None, restype=None):
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)
        del argtypes
        del restype

        # compile core as device function
        args, return_type = sigutils.normalize_signature(sig)
        sig = return_type(*args)

        ocldevfn = ocl.jit(sig, device=True, inline=True)(self.pyfunc)

        # generate outer loop as kernel
        args = ['a%d' % i for i in range(len(sig.args))]
        funcname = self.pyfunc.__name__
        fmts = dict(name=funcname,
                    args=', '.join(args),
                    argitems=', '.join('%s[__tid__]' % i for i in args))
        kernelsource = vectorizer_stager_source % fmts
        glbl = self.pyfunc.__globals__
        glbl.update({'__ocl__': ocl,
                     '__core__': ocldevfn})

        _exec(kernelsource, glbl)

        stager = glbl['__vectorized_%s' % funcname]
        # Force all C contiguous
        kargs = [a[::1] for a in list(sig.args) + [sig.return_type]]
        kernel = ocl.jit(argtypes=kargs)(stager)

        argdtypes = tuple(to_dtype(t) for t in sig.args)
        resdtype = to_dtype(sig.return_type)
        self.kernelmap[tuple(argdtypes)] = resdtype, kernel

    def build_ufunc(self):
        return OclUFuncDispatcher(self.kernelmap)


class OclUFuncDispatcher(object):
    """
    Invoke the Ocl ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels

    def __call__(self, *args, **kws):
        """
        *args: numpy arrays or DeviceArrayBase (created by ocl.to_device).
               Cannot mix the two types in one call.

        **kws:
            stream -- opencl queue; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return OclUFuncCallResolution.call(self.functions, args, kws)


class UFuncCallResolution(object):
    """
    Prepare ufunc arguments for vectorize.
    """
    DEFAULT_STREAM = None
    SUPPORT_DEVICE_SLICING = False

    def __init__(self, typemap, args):
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
            # elif isinstance(arg, (int, longint, float, complex)):
            elif isinstance(arg, (int, float, complex)):
                # Is scalar
                self.scalarpos.append(i)
            else:
                raise TypeError("argument #%d has invalid type" % (i + 1,))

    def _fill_argtypes(self):
        for i, ary in enumerate(self.arrays):
            if ary is not None:
                self.argtypes[i] = ary.dtype

    def _guess_signature(self):
        # Find all matching signature, ignoring missing argtype
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

        self.argtypes = matches[0]

    def _get_actual_args(self):
        for i in self.scalarpos:
            self.arrays[i] = np.array([self.args[i]], dtype=self.argtypes[i])

        return self.arrays

    def _broadcast(self, arys):
        # Get the biggest shape
        nds = [a.ndim for a in arys]
        biggest = np.argmax(nds)
        shape = arys[biggest].shape

        for i, ary in enumerate(arys):
            if ary.shape == shape:
                pass

            elif ary.shape[0] == 1:
                if self.is_device_array(ary):
                    arys[i] = self.broadcast_device(ary)

                else:
                    missingdim = len(shape) - len(ary.shape) + 1
                    strides = (0,) * missingdim + ary.strides[1:]
                    strided = np.lib.stride_tricks.as_strided(ary,
                                                              shape=shape,
                                                              strides=strides)
                    arys[i] = self.force_array_layout(strided)

            else:
                raise ValueError("arg #%d cannot be broadcasted" % (i + 1))

        return arys

    def get_arguments(self):
        self._fill_arrays()
        self._guess_signature()
        arys = self._get_actual_args()
        return self._broadcast(arys)

    def get_function(self):
        return self.typemap[self.argtypes]

    def is_device_array(self, obj):
        return False

    def broadcast_device(self, ary):
        raise NotImplementedError("broadcasting on device is not supported")

    def force_array_layout(self, ary):
        return ary

    @classmethod
    def call(cls, typemap, args, kws):
        # Handle keywords
        stream = kws.pop('stream', cls.DEFAULT_STREAM)
        out = kws.pop('out', None)

        if kws:
            warnings.warn("unrecognized keywords: %s" % ', '.join(kws))

        # Begin call resolution
        cr = cls(typemap, args)
        args = cr.get_arguments()
        resty, func = cr.get_function()

        if args[0].ndim > 1:
            if not cr.SUPPORT_DEVICE_SLICING:
                raise NotImplementedError("Support 1D array only.")
            else:
                raise NotImplementedError

        # Prepare argument on the device
        devarys = []
        all_device = True
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
            else:
                dev_a = cr.to_device(a)
                devarys.append(dev_a)
                all_device = False

        # Launch
        shape = args[0].shape
        if out is None:
            # No output is provided
            devout = cr.device_array(shape, resty)

            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)

            if all_device:
                # If all arguments are on device,
                # Keep output on the device
                return devout
            else:
                # Otherwise, transfer output back to host
                return devout.copy_to_host()

        elif cr.is_device_array(out):
            # If output is provided and it is a device array,
            # Return device array
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout

        else:
            # If output is provided and it is a host array,
            # Return host array
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.device_array(shape, resty)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.copy_to_host(out)

    def to_device(self, hostary):
        raise NotImplementedError

    def device_array(self, shape, dtype):
        raise NotImplementedError

    def launch(self, func, count, stream, args):
        raise NotImplementedError


class OclUFuncCallResolution(UFuncCallResolution):
    DEFAULT_STREAM = None
    ARRAY_ORDER = 'C'

    def is_device_array(self, obj):
        return oclarray.is_ocl_ndarray(obj)

    def to_device(self, hostary):
        return ocl.to_device(hostary)

    def launch(self, func, count, stream, args):
        func.configure(count, stream=stream)(*args)

    def device_array(self, shape, dtype):
        return ocl.device_array(shape=shape, dtype=dtype)

    def force_array_layout(self, ary):
        return np.ascontiguousarray(ary)
