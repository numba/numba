"""
This scripts specifies all PTX special objects.
"""
from __future__ import print_function, absolute_import, division
import operator
import numpy
import llvmlite.llvmpy.core as lc
from numba import types, ir, typing, macro
from .cudadrv import nvvm


class Stub(object):
    '''A stub object to represent special objects which is meaningless
    outside the context of CUDA-python.
    '''
    _description_ = '<ptx special value>'
    __slots__ = () # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_

#-------------------------------------------------------------------------------
# SREG

SREG_SIGNATURE = typing.signature(types.int32)


class threadIdx(Stub):
    '''threadIdx.{x, y, z}
    '''
    _description_ = '<threadIdx.{x,y,z}>'

    x = macro.Macro('tid.x', SREG_SIGNATURE)
    y = macro.Macro('tid.y', SREG_SIGNATURE)
    z = macro.Macro('tid.z', SREG_SIGNATURE)


class blockIdx(Stub):
    '''blockIdx.{x, y}
    '''
    _description_ = '<blockIdx.{x,y,z}>'

    x = macro.Macro('ctaid.x', SREG_SIGNATURE)
    y = macro.Macro('ctaid.y', SREG_SIGNATURE)
    z = macro.Macro('ctaid.z', SREG_SIGNATURE)


class blockDim(Stub):
    '''blockDim.{x, y, z}
    '''
    x = macro.Macro('ntid.x', SREG_SIGNATURE)
    y = macro.Macro('ntid.y', SREG_SIGNATURE)
    z = macro.Macro('ntid.z', SREG_SIGNATURE)


class gridDim(Stub):
    '''gridDim.{x, y}
    '''
    _description_ = '<gridDim.{x,y,z}>'
    x = macro.Macro('nctaid.x', SREG_SIGNATURE)
    y = macro.Macro('nctaid.y', SREG_SIGNATURE)
    z = macro.Macro('nctaid.z', SREG_SIGNATURE)

#-------------------------------------------------------------------------------
# Grid Macro

def _ptx_grid1d(): pass


def _ptx_grid2d(): pass


def grid_expand(ndim):
    """grid(ndim)

    ndim: [int] 1 or 2

        if ndim == 1:
            return cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        elif ndim == 2:
            x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
            return x, y
    """
    if ndim == 1:
        fname = "ptx.grid.1d"
        restype = types.int32
    elif ndim == 2:
        fname = "ptx.grid.2d"
        restype = types.UniTuple(types.int32, 2)
    else:
        raise ValueError('argument can only be 1 or 2')

    return ir.Intrinsic(fname, typing.signature(restype, types.intp),
                        args=[ndim])


grid = macro.Macro('ptx.grid', grid_expand, callable=True)

#-------------------------------------------------------------------------------
# Gridsize Macro

def gridsize_expand(ndim):
    """gridsize(ndim)

    ndim: [int] 1 or 2

        if ndim == 1:
            return cuda.blockDim.x * cuda.gridDim.x
        elif ndim == 2:
            x = cuda.blockDim.x * cuda.gridDim.x
            y = cuda.blockDim.y * cuda.gridDim.y
            return x, y
    """
    if ndim == 1:
        fname = "ptx.gridsize.1d"
        restype = types.int32
    elif ndim == 2:
        fname = "ptx.gridsize.2d"
        restype = types.UniTuple(types.int32, 2)
    else:
        raise ValueError('argument can only be 1 or 2')

    return ir.Intrinsic(fname, typing.signature(restype, types.intp),
                        args=[ndim])


gridsize = macro.Macro('ptx.gridsize', gridsize_expand, callable=True)

#-------------------------------------------------------------------------------
# synthreads

class syncthreads(Stub):
    '''syncthreads()

    Synchronizes all threads in the thread block.
    '''
    _description_ = '<syncthread()>'

#-------------------------------------------------------------------------------
# shared


def shared_array(shape, dtype):
    ndim = 1
    if isinstance(shape, tuple):
        ndim = len(shape)
    elif not isinstance(shape, int):
        raise TypeError("invalid type for shape; got {0}".format(type(shape)))

    fname = "ptx.smem.alloc"
    restype = types.Array(dtype, ndim, 'C')
    if isinstance(shape, int):
        sig = typing.signature(restype, types.intp, types.Any)
    else:
        sig = typing.signature(restype, types.UniTuple(types.intp, ndim),
                               types.Any)

    return ir.Intrinsic(fname, sig, args=(shape, dtype))


class shared(Stub):
    """shared namespace
    """
    _description_ = '<shared>'

    array = macro.Macro('shared.array', shared_array, callable=True,
                        argnames=['shape', 'dtype'])


#-------------------------------------------------------------------------------
# local array


def local_array(shape, dtype):
    ndim = 1
    if isinstance(shape, tuple):
        ndim = len(shape)
    elif not isinstance(shape, int):
        raise TypeError("invalid type for shape; got {0}".format(type(shape)))

    fname = "ptx.lmem.alloc"
    restype = types.Array(dtype, ndim, 'C')
    if isinstance(shape, int):
        sig = typing.signature(restype, types.intp, types.Any)
    else:
        sig = typing.signature(restype, types.UniTuple(types.intp, ndim),
                               types.Any)

    return ir.Intrinsic(fname, sig, args=(shape, dtype))


class local(Stub):
    '''shared namespace
    '''
    _description_ = '<local>'

    array = macro.Macro('local.array', local_array, callable=True,
                        argnames=['shape', 'dtype'])

#-------------------------------------------------------------------------------
# const array


def const_array_like(ndarray):
    fname = "ptx.cmem.arylike"

    from .descriptor import CUDATargetDesc
    aryty = CUDATargetDesc.typingctx.resolve_argument_type(ndarray)

    sig = typing.signature(aryty, aryty)
    return ir.Intrinsic(fname, sig, args=[ndarray])

    raise NotImplementedError
    [aryarg] = args
    ary = aryarg.value
    count = reduce(operator.mul, ary.shape)
    dtype = types.from_dtype(numpy.dtype(ary.dtype))

    def impl(context, args, argtys, retty):
        builder = context.builder
        lmod = builder.basic_block.function.module

        addrspace = nvvm.ADDRSPACE_CONSTANT

        data_t = dtype.llvm_as_value()

        flat = ary.flatten(order='A')  # preserve order
        constvals = [dtype.llvm_const(flat[i]) for i in range(flat.size)]
        constary = lc.Constant.array(data_t, constvals)

        gv = lmod.add_global_variable(constary.type, "cmem", addrspace)
        gv.linkage = lc.LINKAGE_INTERNAL
        gv.global_constant = True
        gv.initializer = constary

        byte = lc.Type.int(8)
        byte_ptr_as = lc.Type.pointer(byte, addrspace)
        to_generic = nvvmutils.insert_addrspace_conv(lmod, byte, addrspace)
        rawdata = builder.call(to_generic, [builder.bitcast(gv, byte_ptr_as)])
        data = builder.bitcast(rawdata, lc.Type.pointer(data_t))

        llintp = types.intp.llvm_as_value()
        cshape = lc.Constant.array(llintp,
                                   map(types.const_intp, ary.shape))

        cstrides = lc.Constant.array(llintp,
                                     map(types.const_intp, ary.strides))
        res = lc.Constant.struct([lc.Constant.null(data.type), cshape,
                                  cstrides])
        res = builder.insert_value(res, data, 0)
        return res

    if ary.flags['C_CONTIGUOUS']:
        contig = 'C'
    elif ary.flags['F_CONTIGUOUS']:
        contig = 'F'
    else:
        raise TypeError("array must be either C/F contiguous to be used as a "
                        "constant")

    impl.codegen = True
    impl.return_type = types.arraytype(dtype, ary.ndim, 'A')
    return impl


class const(Stub):
    '''shared namespace
    '''
    _description_ = '<const>'

    array_like = macro.Macro('const.array_like', const_array_like,
                             callable=True, argnames=['ary'])

#-------------------------------------------------------------------------------
# atomic

class atomic(Stub):
    """atomic namespace
    """
    _description_ = '<atomic>'

    class add(Stub):
        """add(ary, idx, val)

        Perform atomic ary[idx] += val
        """

