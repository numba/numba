from __future__ import print_function, absolute_import
import warnings
import numpy as np
from numba import sigutils, ocl
from numba.utils import IS_PY3
from numba.ocl.ocldrv import oclarray
from numba.npyufunc.deviceufunc import UFuncMechanism

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


def _to_dtype(ty):
    return np.dtype(str(ty))


class OclVectorize(object):
    """OpenCL ufunc builder
    """
    def __init__(self, func, targetoptions={}):
        assert not targetoptions
        self.pyfunc = func
        self.kernelmap = {}  # { arg_dtype: (return_dtype), kernel }

    def add(self, sig=None, argtypes=None, restype=None):
        """Add signature
        """
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)

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

        argdtypes = tuple(_to_dtype(t) for t in sig.args)
        resdtype = _to_dtype(sig.return_type)
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
        *args: numpy arrays or device array (created by ocl.to_device).
               Cannot mix the two types in one call.

        **kws:
            stream -- opencl queue; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return OclUFuncMechanism.call(self.functions, args, kws)


class OclUFuncMechanism(UFuncMechanism):
    """
    Provide OpenCL specialization
    """
    DEFAULT_STREAM = None
    ARRAY_ORDER = 'C'   # SPIR 1.2 does not allow inttoptr used in any layout

    def is_device_array(self, obj):
        return oclarray.is_ocl_ndarray(obj)

    def to_device(self, hostary):
        return ocl.to_device(hostary)

    def launch(self, func, count, stream, args):
        func.configure(count, stream=stream)(*args)

    def device_array(self, shape, dtype):
        return ocl.device_array(shape=shape, dtype=dtype)

    def force_array_layout(self, ary):
        """
        All arrays must be C contiguous
        """
        return np.ascontiguousarray(ary)
