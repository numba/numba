import operator
import warnings
import numpy
from numba.npyufunc.sigparse import parse_signature
from numba import sigutils
import llvm.core as lc
from numbapro.npm import types, cgutils, aryutils
from numbapro.cudapy.execution import CUDAKernel
from numbapro.cudapy import ptx, compiler
from . import dispatch

vectorizer_stager_source = '''
def __vectorized_%(name)s(%(args)s, __out__):
    __tid__ = __cuda__.grid(1)
    __out__[__tid__] = __core__(%(argitems)s)
'''

def to_dtype(ty):
    return numpy.dtype(str(ty))


class CudaVectorize(object):
    def __init__(self, func, targetoptions={}):
        assert not targetoptions
        self.pyfunc = func
        self.kernelmap = {} # { arg_dtype: (return_dtype), cudakernel }

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
        from numbapro import cuda
        # compile core as device function
        args, return_type = sigutils.normalize_signature(sig)
        sig = return_type(*args)

        cudevfn = cuda.jit(sig,
                           device=True, inline=True)(self.pyfunc)

        # generate outer loop as kernel
        args = ['a%d' % i for i in range(len(sig.args))]
        funcname = self.pyfunc.__name__
        fmts = dict(name=funcname,
                    args = ', '.join(args),
                    argitems = ', '.join('%s[__tid__]' % i for i in args))
        kernelsource = vectorizer_stager_source % fmts
        glbl = self.pyfunc.func_globals
        glbl.update({'__cuda__': cuda,
                     '__core__': cudevfn})
        exec kernelsource in glbl

        stager = glbl['__vectorized_%s' % funcname]
        kargs = [a[:] for a in list(sig.args) + [sig.return_type]]
        kernel = cuda.jit(argtypes=kargs)(stager)

        argdtypes = tuple(to_dtype(t) for t in sig.args)
        resdtype = to_dtype(sig.return_type)
        self.kernelmap[tuple(argdtypes)] = resdtype, kernel

    def build_ufunc(self):
        return dispatch.CudaUFuncDispatcher(self.kernelmap)

#------------------------------------------------------------------------------
# Generalized CUDA ufuncs

class CudaGUFuncVectorize(object):

    def __init__(self, func, sig, targetoptions={}):
        assert not targetoptions
        self.pyfunc = func
        self.signature = sig
        self.inputsig, self.outputsig = parse_signature(self.signature)
        assert len(self.outputsig) == 1, "only support 1 output"
        self.kernelmap = {}  # { arg_dtype: (return_dtype), cudakernel }

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
        from numbapro import cuda
        cudevfn = cuda.jit(sig,
                           device=True, inline=True)(self.pyfunc)

        dims = [len(x) for x in self.inputsig]
        dims += [len(x) for x in self.outputsig]
        lmod, lgufunc, outertys, excs = build_gufunc_stager(cudevfn, dims)

        wrapper = compiler.generate_kernel_wrapper(lgufunc, bool(excs))
        ptxcode = compiler.to_ptx(wrapper)
        kernel = CUDAKernel(wrapper.name, ptxcode, outertys, excs=excs)
        kernel.bind()

        dtypes = tuple(numpy.dtype(str(t.desc.element)) for t in outertys)
        self.kernelmap[tuple(dtypes[:-1])] = dtypes[-1], kernel
        
    def build_ufunc(self):
        engine = GUFuncEngine(self.inputsig, self.outputsig)
        return dispatch.CUDAGenerializedUFunc(kernelmap=self.kernelmap,
                                              engine=engine)

def build_gufunc_stager(devfn, dims):
    lmod, lfunc, return_type, args, excs = devfn._npm_context_
    assert return_type == types.void, "must return nothing"

    outer_args = []
    for arg, dim in zip(args, dims):
        if not isinstance(arg.desc, types.Array):
            typ = types.arraytype(arg, 1, 'A')
        else:
            typ = types.arraytype(arg.desc.element, dim + 1, arg.desc.order)
        outer_args.append(typ)

    # copy a new module
    lmod = lmod.clone()
    lfunc = lmod.get_function_named(lfunc.name)

    argtypes = [t.llvm_as_argument() for t in outer_args]
    fnty = lc.Type.function(lc.Type.int(), argtypes)
    lgufunc = lmod.add_function(fnty, name='gufunc_%s' % lfunc.name)

    builder = lc.Builder.new(lgufunc.append_basic_block(''))

    # allocate new array with one less dimension
    
    fname_tidx = ptx.SREG_MAPPING[ptx._ptx_sreg_tidx]
    fname_ntidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ntidx]
    fname_ctaidx = ptx.SREG_MAPPING[ptx._ptx_sreg_ctaidx]

    li32 = types.uint32.llvm_as_value()
    fn_tidx = cgutils.get_function(builder, fname_tidx, li32, ())
    fn_ntidx = cgutils.get_function(builder, fname_ntidx, li32, ())
    fn_ctaidx = cgutils.get_function(builder, fname_ctaidx, li32, ())

    tidx = builder.call(fn_tidx, ())
    ntidx = builder.call(fn_ntidx, ())
    ctaidx = builder.call(fn_ctaidx, ())

    tx = types.uint32.llvm_cast(builder, tidx, types.intp)
    bw = types.uint32.llvm_cast(builder, ntidx, types.intp)
    bx = types.uint32.llvm_cast(builder, ctaidx, types.intp)

    tid = builder.add(tx, builder.mul(bw, bx))

    arguments = []
    for aryptr, inner, outer, dim in zip(lgufunc.args, args, outer_args, dims):
        
        ary = builder.load(aryptr)
        data = aryutils.getdata(builder, ary)
        shape = aryutils.getshape(builder, ary)
        strides = aryutils.getstrides(builder, ary)

        slice_data = get_slice_data(builder, data, shape, strides,
                                    outer.desc.order, tid)

        if not isinstance(inner.desc, types.Array): # scalar argument
            ok = builder.icmp(lc.ICMP_SLT, tid, shape[0])
            index = builder.select(ok, tid, types.intp.llvm_const(0))
            item = aryutils.getitem(builder, ary, order=outer.desc.order,
                                    indices=[index])

            arguments.append(item)
        else:
            storage = builder.alloca(inner.llvm_as_value())
            arguments.append(storage)

            if outer.desc.ndim == 1:
                item = aryutils.ndarray(builder,
                                          dtype=outer.desc.element,
                                          ndim=inner.desc.ndim,
                                          order=inner.desc.order,
                                          shape=[types.intp.llvm_const(1)],
                                          strides=[types.intp.llvm_const(0)],
                                          data=slice_data)
            else:
                item = aryutils.ndarray(builder,
                                          dtype=outer.desc.element,
                                          ndim=inner.desc.ndim,
                                          order=inner.desc.order,
                                          shape=shape[1:],
                                          strides=strides[1:],
                                          data=slice_data)

            builder.store(item, storage)


    errcode = builder.call(lfunc, arguments)
    builder.ret(errcode)

    lmod.verify()
    return lmod, lgufunc, outer_args, excs


def get_slice_data(builder, data, shape, strides, order, index):
    intp = shape[0].type
    indices = [builder.zext(index, intp)]
    indices += [lc.Constant.null(intp) for i in range(len(shape) - 1)]
    return aryutils.getpointer(builder, data, shape, strides, order, indices)


class GUFuncEngine(object):
    '''Determine how to broadcast and execute a gufunc
    base on input shape and signature
    '''
    @classmethod
    def from_signature(cls, signature):
        return cls(*parse_signature(signature))

    def __init__(self, inputsig, outputsig):
        # signatures
        self.sin = inputsig
        self.sout = outputsig
        # argument count
        self.nin = len(self.sin)
        self.nout = len(self.sout)

    def schedule(self, ishapes):
        if len(ishapes) != self.nin:
            raise TypeError('invalid number of input argument')

        # associate symbol values for input signature
        symbolmap = {}
        outer_shapes = []
        inner_shapes = []

        for argn, (shape, symbols) in enumerate(zip(ishapes, self.sin)):
            argn += 1 # start from 1 for human
            inner_ndim = len(symbols)
            if len(shape) < inner_ndim:
                fmt = "arg #%d: insufficient inner dimension"
                raise ValueError(fmt % (argn,))
            if inner_ndim:
                inner_shape = shape[-inner_ndim:]
                outer_shape = shape[:-inner_ndim]
            else:
                inner_shape = ()
                outer_shape = shape


            for axis, (dim, sym) in enumerate(zip(inner_shape, symbols)):
                axis += len(outer_shape)
                if sym in symbolmap:
                    if symbolmap[sym] != dim:
                        fmt = "arg #%d: shape[%d] mismatch argument"
                        raise ValueError(fmt % (argn, axis))
                symbolmap[sym] = dim

            outer_shapes.append(outer_shape)
            inner_shapes.append(inner_shape)
    
        # solve output shape
        oshapes = []
        for outsig in self.sout:
            oshape = []
            for sym in outsig:
                oshape.append(symbolmap[sym])
            oshapes.append(tuple(oshape))

        # find the biggest outershape as looping dimension
        sizes = [reduce(operator.mul, s, 1) for s in outer_shapes]
        largest_i = numpy.argmax(sizes)
        loopdims = outer_shapes[largest_i]

        pinned = [False] * self.nin          # same argument for each iteration
        for i, d in enumerate(outer_shapes):
            if d != loopdims:
                if d == (1,) or d == ():
                    pinned[i] = True
                else:
                    fmt = "arg #%d: outer dimension mismatch"
                    raise ValueError(fmt % (i + 1,))

        return GUFuncSchedule(self, inner_shapes, oshapes, loopdims, pinned)

class GUFuncSchedule(object):
    def __init__(self, parent, ishapes, oshapes, loopdims, pinned):
        self.parent = parent
        # core shapes
        self.ishapes = ishapes
        self.oshapes = oshapes
        # looping dimension
        self.loopdims = loopdims
        self.loopn = reduce(operator.mul, loopdims, 1)
        # flags
        self.pinned = pinned

        self.output_shapes = [loopdims + s for s in oshapes]

    def __str__(self):
        import pprint
        attrs = 'ishapes', 'oshapes', 'loopdims', 'loop' 'pinned'
        values = [(k, getattr(self, k)) for k in attrs]
        return pprint.pformat(dict(values))
