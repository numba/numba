import re
import llvm.core as lc
from numbapro import cuda
from numbapro.npm import types, cgutils, aryutils
from numbapro.cudapy.compiler import to_ptx
from numbapro.cudapy.execution import CUDAKernel
from numbapro.cudapy import ptx
from . import dispatch
from numbapro.vectorizers._common import parse_signature

vectorizer_stager_source = '''
def __vectorized_%(name)s(%(args)s, __out__):
    __tid__ = __cuda__.grid(1)
    __out__[__tid__] = __core__(%(argitems)s)
'''

class CudaVectorize(object):
    def __init__(self, func):
        self.pyfunc = func
        self.kernelmap = {} # { arg_dtype: (return_dtype), cudakernel }

    def add(self, restype, argtypes):
        # compile core as device function
        cudevfn = cuda.jit(restype, argtypes,
                           device=True, inline=True)(self.pyfunc)

        # generate outer loop as kernel
        args = ['a%d' % i for i in range(len(argtypes))]
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
        kargs = [a[:] for a in list(argtypes) + [restype]]
        kernel = cuda.jit(argtypes=kargs)(stager)

        argdtypes = tuple(t.get_dtype() for t in argtypes)
        resdtype = restype.get_dtype()
        self.kernelmap[tuple(argdtypes)] = resdtype, kernel

    def build_ufunc(self):
        return dispatch.CudaUFuncDispatcher(self.kernelmap)

#------------------------------------------------------------------------------
# Generalized CUDA ufuncs

class CudaGUFuncVectorize(object):

    def __init__(self, func, sig):
        self.pyfunc = func
        self.signature = sig
        self.inputsig, self.outputsig = parse_signature(self.signature)
        assert len(self.outputsig) == 1, "only support 1 output"
        self.kernelmap = {}  # { arg_dtype: (return_dtype), cudakernel }

    def add(self, argtypes, restype=None):
        cudevfn = cuda.jit(argtypes=argtypes,
                           device=True, inline=True)(self.pyfunc)

        dims = [len(x) for x in self.inputsig]
        dims += [len(x) for x in self.outputsig]
        lmod, lgufunc, outertys = build_gufunc_stager(cudevfn, dims)

        ptx = to_ptx(lgufunc)
        kernel = CUDAKernel(lgufunc.name, ptx, outertys, excs=None)
        kernel.bind()

        dtypes = tuple(t.dtype.get_dtype() for t in argtypes)
        self.kernelmap[tuple(dtypes[:-1])] = dtypes[-1], kernel
        
    def build_ufunc(self):
        return dispatch.CudaGUFuncDispatcher(self.kernelmap, self.signature)

def build_gufunc_stager(devfn, dims):
    lmod, lfunc, return_type, args, excs = devfn._npm_context_
    assert return_type is None, "must return nothing"
    outer_args = [types.arraytype(a.desc.element, dim + 1, a.desc.order)
                  for a, dim in zip(args, dims)]

    # copy a new module
    lmod = lmod.clone()
    lfunc = lmod.get_function_named(lfunc.name)

    argtypes = [t.llvm_as_argument() for t in outer_args]
    fnty = lc.Type.function(lc.Type.void(), argtypes)
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

    slices = []
    for aryptr, inner, outer in zip(lgufunc.args, args,
                                  outer_args):
        slice = builder.alloca(inner.llvm_as_value())
        slices.append(slice)

        ary = builder.load(aryptr)
        data = aryutils.getdata(builder, ary)
        shape = aryutils.getshape(builder, ary)
        strides = aryutils.getstrides(builder, ary)

        slice_data = get_slice_data(builder, data, shape, strides,
                                    outer.desc.order, tid)

        if outer.desc.ndim == 1:
            newary = aryutils.ndarray(builder,
                                      dtype=outer.desc.element,
                                      ndim=inner.desc.ndim,
                                      order=inner.desc.order,
                                      shape=[types.intp.llvm_const(1)],
                                      strides=[types.intp.llvm_const(0)],
                                      data=slice_data)
        else:
            newary = aryutils.ndarray(builder,
            dtype=outer.desc.element,
                                      ndim=inner.desc.ndim,
                                      order=inner.desc.order,
                                      shape=shape[1:],
                                      strides=strides[1:],
                                      data=slice_data)

        builder.store(newary, slice)

    builder.call(lfunc, slices)
    builder.ret_void()

    lmod.verify()
    return lmod, lgufunc, outer_args


def get_slice_data(builder, data, shape, strides, order, index):
    intp = shape[0].type
    indices = [builder.zext(index, intp)]
    indices += [lc.Constant.null(intp) for i in range(len(shape) - 1)]
    return aryutils.getpointer(builder, data, shape, strides, order, indices)
