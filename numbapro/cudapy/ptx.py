'''
This scripts specifies all PTX special objects.
'''
import operator
import numpy
import llvm.core as lc
from numbapro.npm import types, macro, symbolic
from numbapro.cudadrv import nvvm
from numbapro._utils.mviewbuf import memoryview_get_extents
from . import nvvmutils

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

def _ptx_sreg_tidx(): pass
def _ptx_sreg_tidy(): pass
def _ptx_sreg_tidz(): pass

def _ptx_sreg_ntidx(): pass
def _ptx_sreg_ntidy(): pass
def _ptx_sreg_ntidz(): pass

def _ptx_sreg_ctaidx(): pass
def _ptx_sreg_ctaidy(): pass

def _ptx_sreg_nctaidx(): pass
def _ptx_sreg_nctaidy(): pass


class threadIdx(Stub):
    '''threadIdx.{x, y, z}
    '''
    _description_ = '<threadIdx.{x,y,z}>'

    x = macro.Macro('threadIdx.x', _ptx_sreg_tidx)
    y = macro.Macro('threadIdx.y', _ptx_sreg_tidy)
    z = macro.Macro('threadIdx.z', _ptx_sreg_tidz)

class blockIdx(Stub):
    '''blockIdx.{x, y}
    '''
    _description_ = '<blockIdx.{x,y}>'

    x = macro.Macro('blockIdx.x', _ptx_sreg_ctaidx)
    y = macro.Macro('blockIdx.y', _ptx_sreg_ctaidy)

class blockDim(Stub):
    '''blockDim.{x, y, z}
    '''
    x = macro.Macro('blockDim.x', _ptx_sreg_ntidx)
    y = macro.Macro('blockDim.y', _ptx_sreg_ntidy)
    z = macro.Macro('blockDim.z', _ptx_sreg_ntidz)

class gridDim(Stub):
    '''gridDim.{x, y}
    '''
    _description_ = '<gridDim.{x,y}>'
    x = macro.Macro('gridDim.x', _ptx_sreg_nctaidx)
    y = macro.Macro('gridDim.y', _ptx_sreg_nctaidy)

SREG_MAPPING = {
    _ptx_sreg_tidx: 'llvm.nvvm.read.ptx.sreg.tid.x',
    _ptx_sreg_tidy: 'llvm.nvvm.read.ptx.sreg.tid.y',
    _ptx_sreg_tidz: 'llvm.nvvm.read.ptx.sreg.tid.z',
    
    _ptx_sreg_ntidx: 'llvm.nvvm.read.ptx.sreg.ntid.x',
    _ptx_sreg_ntidy: 'llvm.nvvm.read.ptx.sreg.ntid.y',
    _ptx_sreg_ntidz: 'llvm.nvvm.read.ptx.sreg.ntid.z',
    
    _ptx_sreg_ctaidx: 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    _ptx_sreg_ctaidy: 'llvm.nvvm.read.ptx.sreg.ctaid.y',
    
    _ptx_sreg_nctaidx: 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    _ptx_sreg_nctaidy: 'llvm.nvvm.read.ptx.sreg.nctaid.y',
}

SREG_FUNCTION_TYPE = lc.Type.function(lc.Type.int(), [])
SREG_TYPE = types.uint32

#-------------------------------------------------------------------------------
# Grid Macro

def _ptx_grid1d(): pass

def _ptx_grid2d(): pass

def grid_expand(args):
    '''grid(ndim)

    ndim: [int] 1 or 2
    
        if ndim == 1:
            return cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        elif ndim == 2:
            x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
            return x, y
    '''
    if len(args) != 1:
        raise ValueError('takes exactly 1 argument')
    ndim, = args

    try:
        ndim = symbolic.const_value_from_inst(ndim)
    except ValueError:
        raise ValueError('argument must be constant')

    if ndim == 1:
        return _ptx_grid1d
    elif ndim == 2:
        return _ptx_grid2d
    else:
        raise ValueError('argument can only be 1 or 2')


grid = macro.Macro('grid', grid_expand, callable=True)

#-------------------------------------------------------------------------------
# synthreads

class syncthreads(Stub):
    '''syncthreads() 
    
    Synchronizes all threads in the thread block.
    '''
    _description_ = '<syncthread()>'


#-------------------------------------------------------------------------------
# generic array


def _generic_array(args, symbol_name, addrspace, can_dynsized=False):
    if len(args) != 2:
        raise ValueError('takes exactly 2 arguments')
    shape, dtype = args

    try:
        shape = symbolic.const_value_from_inst(shape)
    except ValueError:
        raise ValueError('expecting constant value for shape')
    dtype = types.from_numba_type(dtype.value)


    if isinstance(shape, (tuple, list)):
        shape = shape
    elif isinstance(shape, int):
        shape = [shape]
    else:
        raise TypeError('invalid type for shape')

    def impl(context, args, argtys, retty):
        builder = context.builder
        size = reduce(operator.mul, shape)

        elemtype = dtype.llvm_as_value()
        mem_elemtype = retty.desc.element.llvm_as_value()
        mem_type = lc.Type.array(mem_elemtype, size)
        lmod = context.builder.basic_block.function.module
        mem = lmod.add_global_variable(mem_type, symbol_name, addrspace)

        if size == 0:
            if can_dynsized:    # dynamic shared memory
                mem.linkage = lc.LINKAGE_EXTERNAL
            else:
                raise ValueError("zero array length")
        else:            # static shared memory
            mem.linkage = lc.LINKAGE_INTERNAL
            mem.initializer = lc.Constant.undef(mem_type)

        byte = lc.Type.int(8)
        byte_ptr_as = lc.Type.pointer(byte, addrspace)

        to_generic = nvvmutils.insert_addrspace_conv(lmod, byte, addrspace)

        rawdata = builder.call(to_generic,
                               [builder.bitcast(mem, byte_ptr_as)])
        data = builder.bitcast(rawdata, lc.Type.pointer(elemtype))

        llintp = types.intp.llvm_as_value()
        cshape = lc.Constant.array(llintp,
                                   map(types.const_intp, shape))

        strides_raw = [reduce(operator.mul, shape[i + 1:], 1)
                       for i in range(len(shape))]

        strides = [builder.mul(types.sizeof(elemtype), types.const_intp(s))
                   for s in strides_raw]

        cstrides = lc.Constant.array(llintp, strides)

        ary = lc.Constant.struct([lc.Constant.null(data.type), cshape,
                                  cstrides])
        ary = builder.insert_value(ary, data, 0)

        return ary

    impl.codegen = True
    impl.return_type = types.arraytype(dtype, len(shape), 'C')
    return impl

#-------------------------------------------------------------------------------
# shared

def shared_array(args):
    return _generic_array(args, 'smem', nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)

class shared(Stub):
    '''shared namespace
    '''
    _description_ = '<shared>'

    array = macro.Macro('shared.array', shared_array, callable=True,
                        argnames=['shape', 'dtype'])


#-------------------------------------------------------------------------------
# local array


def local_array(args):
    return _generic_array(args, 'lmem', nvvm.ADDRSPACE_LOCAL)


class local(Stub):
    '''shared namespace
    '''
    _description_ = '<local>'

    array = macro.Macro('local.array', local_array, callable=True,
                        argnames=['shape', 'dtype'])

#-------------------------------------------------------------------------------
# const array


def const_array_like(args):
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
    #return _generic_array(args, 'cmem', nvvm.ADDRSPACE_CONSTANT)


class const(Stub):
    '''shared namespace
    '''
    _description_ = '<const>'

    array_like = macro.Macro('const.array_like', const_array_like,
                             callable=True, argnames=['ary'])

#-------------------------------------------------------------------------------
# atomic

class atomic(Stub):
    '''atomic namespace
    '''
    _description_ = '<atomic>'

    class add(Stub):
        '''add(ary, idx, val)
        
        Perform atomic ary[idx] += val
        '''

__all__ = '''
threadIdx
blockIdx
blockDim
gridDim
grid
syncthreads
shared
local
const
atomic
'''.split()
