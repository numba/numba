import numpy
import numba
from numbapro.npm import types
from numbapro.cudapy import compiler
from numbapro.cudapy.execution import CUDAKernelBase
from numbapro.cudadrv import devicearray

NUMBA_TO_NPM_TYPES = {
    numba.int8: types.int8,
    numba.int16: types.int16,
    numba.int32: types.int32,
    numba.int64: types.int64,

    numba.uint8: types.uint8,
    numba.uint16: types.uint16,
    numba.uint32: types.uint32,
    numba.uint64: types.uint64,

    numba.float32: types.float32,
    numba.float64: types.float64,

    numba.complex64: types.complex64,
    numba.complex128: types.complex128,

    numba.void: None,
}

def _eval_type_string(text):
    func = eval(text, vars(numba))
    assert func.is_function, "not a function signature"
    return func.return_type, func.args

def _map_numba_to_npm_types(typ):
    if typ.is_array:
        elem = _map_numba_to_npm_types(typ.dtype)
        ndim = typ.ndim
        order = 'A'
        if typ.is_c_contig:
            order = 'C'
        elif typ.is_f_contig:
            order = 'F'
        return types.arraytype(elem, ndim, order)
    return NUMBA_TO_NPM_TYPES[typ]

def jit(restype=None, argtypes=None, device=False, inline=False, bind=True,
        **kws):
    # eval type string
    if isinstance(restype, str):
        restype, argtypes = _eval_type_string(restype)

    if argtypes is None:
        # must be a function then
        assert restype.is_function, "%s is not a function" % restype
        argtypes = restype.args
        restype = restype.return_type

    # convert Numba types to NPM types
    try:
        restype = (restype
                   if restype is None
                   else _map_numba_to_npm_types(restype))
        argtypes = [_map_numba_to_npm_types(t) for t in argtypes]
    except KeyError, e:
        raise TypeError("invalid type for CUDA: %s" % e)

    if restype and not device:
        raise TypeError("CUDA kernel must have void return type.")

    def kernel_jit(func):
        cukern = compiler.compile_kernel(func, argtypes)
        if bind: cukern.bind()
        return cukern

    def device_jit(func):
        return compiler.compile_device(func, restype, argtypes,
                                            inline=True)

    if device:
        return device_jit
    else:
        return kernel_jit

def autojit(func, **kws):
    return AutoJitCUDAKernel(func, bind=True)

class AutoJitCUDAKernel(CUDAKernelBase):
    def __init__(self, func, bind):
        super(AutoJitCUDAKernel, self).__init__()
        self.intp = {4: types.int32, 8: types.int64}[tuple.__itemsize__]
        self.func = func
        self.bind = bind
        self.definitions = {}

    def __call__(self, *args, **kws):
        argtypes = []
        for a in args:
            if (devicearray.is_cuda_ndarray(a) or
                    isinstance(a, numpy.ndarray)):
                dty = numpy.dtype(a.dtype)
                ety = FROM_DTYPE[dty]
                aty = types.arraytype(ety, a.ndim, 'A')
                argtypes.append(aty)
            elif isinstance(a, complex):
                argtypes.append(types.complex128)
            elif isinstance(a, float):
                argtypes.append(types.float64)
            elif isinstance(a, (int, long)):
                argtypes.append(self.intp)
            else:
                raise TypeError("unsupported type: %s" % type(a))
        sig = tuple(argtypes)
        if sig in self.definitions:
            defn = self.definitions[sig]
        else:
            defn = compiler.compile_kernel(self.func, argtypes)
            if self.bind: defn.bind()

            self.definitions[sig] = defn

        cfg = defn[self.griddim, self.blockdim, self.stream, self.sharedmem]
        cfg(*args, **kws)


FROM_DTYPE = {
    numpy.dtype('int8'): types.int8,
    numpy.dtype('int16'): types.int16,
    numpy.dtype('int32'): types.int32,
    numpy.dtype('int64'): types.int64,

    numpy.dtype('uint8'): types.uint8,
    numpy.dtype('uint16'): types.uint16,
    numpy.dtype('uint32'): types.uint32,
    numpy.dtype('uint64'): types.uint64,

    numpy.dtype('float32'): types.float32,
    numpy.dtype('float64'): types.float64,

    numpy.dtype('complex64'): types.complex64,
    numpy.dtype('complex128'): types.complex128,

}

