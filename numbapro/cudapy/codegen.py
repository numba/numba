import operator
import numpy as np
from llvm.core import Type, Constant, LINKAGE_EXTERNAL, LINKAGE_INTERNAL

from numbapro.npm.errors import CompileError
from numbapro.npm import types
from . import ptx, libdevice
from numbapro.cudadrv.nvvm import ADDRSPACE_SHARED
import numbapro

class CudaPyCGError(CompileError):
    def __init__(self, value, msg):
        super(CudaPyCGError, self).__init__(value, msg)


def declare_sreg(cg, sregobj):
    fname = ptx.SREG_MAPPING[sregobj]
    fnty = ptx.SREG_FUNCTION_TYPE
    func = cg.lmod.get_or_insert_function(fnty, name=fname)
    return func

def cg_sreg(cg, value):
    ty = cg.typemap[value]
    func = declare_sreg(cg, value.args.value)
    res = cg.builder.call(func, ())
    return cg.do_cast(res, ptx.SREG_TYPE, ty)

def cg_grid_macro(cg, value):
    ty = cg.typemap[value]
    assert len(value.args.args) == 1
    arg = value.args.args[0].value
    assert arg.kind == 'Const'
    ndim = arg.args.value
    assert ndim in (1, 2)
    if ndim == 1:
        tx = cg.builder.call(declare_sreg(cg, ptx.threadIdx.x), ())
        bx = cg.builder.call(declare_sreg(cg, ptx.blockIdx.x), ())
        bw = cg.builder.call(declare_sreg(cg, ptx.blockDim.x), ())
        tid = cg.builder.add(tx, cg.builder.mul(bx, bw))
        cg.valmap[value] = cg.do_cast(tid, ptx.SREG_TYPE, ty)
    else:
        assert ndim == 2
        tx = cg.builder.call(declare_sreg(cg, ptx.threadIdx.x), ())
        bx = cg.builder.call(declare_sreg(cg, ptx.blockIdx.x), ())
        bwx = cg.builder.call(declare_sreg(cg, ptx.blockDim.x), ())
        ty = cg.builder.call(declare_sreg(cg, ptx.threadIdx.y), ())
        by = cg.builder.call(declare_sreg(cg, ptx.blockIdx.y), ())
        bwy = cg.builder.call(declare_sreg(cg, ptx.blockDim.y), ())

        tidx = cg.builder.add(tx, cg.builder.mul(bx, bwx))
        tidy = cg.builder.add(ty, cg.builder.mul(by, bwy))

        cg.valmap[value] = tidx, tidy

def cg_syncthreads(cg, value):
    assert value not in cg.typemap, "syncthread() should have no return type"
    assert not value.args.args, "syncthread() takes no argument"
    fname = 'llvm.nvvm.barrier0'
    fnty = Type.function(Type.void(), ())
    func = cg.lmod.get_or_insert_function(fnty, name=fname)
    cg.builder.call(func, ())

def cg_dtype(cg, value):
    pass # yup, no-op

def cg_numba_cast(outty):
    def _cast(cg, value):
        cg.valmap[value] = cg.cast(value.args.args[0].value, outty)
    return _cast

def cg_shared_array(cg, value):
    args = value.args.args

    arytype = cg.typemap[value]
    elemtype = arytype.element

    shape, _dtype = args

    size = reduce(operator.mul, shape)

    smem_elemtype = cg.to_llvm(elemtype)
    smem_type = Type.array(smem_elemtype, size)

    smem = cg.lmod.add_global_variable(smem_type, 'smem', ADDRSPACE_SHARED)

    if size == 0: # dynamic shared memory
        smem.linkage = LINKAGE_EXTERNAL
    else:
        smem.linkage = LINKAGE_INTERNAL
        smem.initializer = Constant.undef(smem_type)

    smem_elem_ptr_ty = Type.pointer(smem_elemtype)
    smem_elem_ptr_ty_addrspace = Type.pointer(smem_elemtype, ADDRSPACE_SHARED)

    # convert to generic addrspace
    tyname = str(smem_elemtype)
    tyname = {'float': 'f32', 'double': 'f64'}.get(tyname, tyname)
    s2g_name_fmt = 'llvm.nvvm.ptr.shared.to.gen.p0%s.p%d%s'
    s2g_name = s2g_name_fmt % (tyname, ADDRSPACE_SHARED, tyname)
    s2g_fnty = Type.function(smem_elem_ptr_ty, [smem_elem_ptr_ty_addrspace])
    shared_to_generic = cg.lmod.get_or_insert_function(s2g_fnty, s2g_name)

    data = cg.builder.call(shared_to_generic,
                        [cg.builder.bitcast(smem, smem_elem_ptr_ty_addrspace)])

    def const_intp(x):
        return Constant.int_signextend(cg.typesetter.llvm_intp, x)

    cshape = Constant.array(cg.typesetter.llvm_intp, map(const_intp, shape))

    strides_raw = [reduce(operator.mul, shape[i + 1:], 1)
                   for i in range(len(shape))]
    strides = [cg.builder.mul(cg.sizeof(cg.typesetter.intp), const_intp(s))
               for s in strides_raw]

    cstrides = Constant.array(cg.typesetter.llvm_intp, strides)

    ary = Constant.struct([Constant.null(data.type), cshape, cstrides])
    ary = cg.builder.insert_value(ary, data, 0)

    aryptr = cg.builder.alloca(ary.type)
    cg.builder.store(ary, aryptr)

    cg.valmap[value] = aryptr


#-------------------------------------------------------------------------------

cudapy_global_codegen_ext = {
    ptx.threadIdx.x:    cg_sreg,
    ptx.threadIdx.y:    cg_sreg,
    ptx.threadIdx.z:    cg_sreg,
    ptx.blockIdx.x:     cg_sreg,
    ptx.blockIdx.y:     cg_sreg,

    ptx.blockDim.x:     cg_sreg,
    ptx.blockDim.y:     cg_sreg,
    ptx.blockDim.z:     cg_sreg,
    ptx.gridDim.x:      cg_sreg,
    ptx.gridDim.y:      cg_sreg,
}

np_dtype_global_ext = {
    np.dtype(np.int8):      	cg_dtype,
    np.dtype(np.int16):     	cg_dtype,
    np.dtype(np.int32):     	cg_dtype,
    np.dtype(np.int64):         cg_dtype,
    np.dtype(np.uint8):         cg_dtype,
    np.dtype(np.uint16):    	cg_dtype,
    np.dtype(np.uint32):    	cg_dtype,
    np.dtype(np.uint64):    	cg_dtype,
    np.dtype(np.float32):       cg_dtype,
    np.dtype(np.float64):       cg_dtype,
    np.dtype(np.complex64):     cg_dtype,
    np.dtype(np.complex128):    cg_dtype,
}

cudapy_global_codegen_ext.update(np_dtype_global_ext)

cudapy_call_codegen_ext = {
    ptx.grid:           cg_grid_macro,
    ptx.syncthreads:    cg_syncthreads,
    ptx.shared.array:   cg_shared_array,
}

numba_cast_ext = {
    numbapro.int8:          cg_numba_cast(types.int8),
    numbapro.int16:         cg_numba_cast(types.int16),
    numbapro.int32:         cg_numba_cast(types.int32),
    numbapro.int64:         cg_numba_cast(types.int64),
    numbapro.uint8:         cg_numba_cast(types.uint8),
    numbapro.uint16:        cg_numba_cast(types.uint16),
    numbapro.uint32:        cg_numba_cast(types.uint32),
    numbapro.uint64:        cg_numba_cast(types.uint64),
    numbapro.float32:       cg_numba_cast(types.float32),
    numbapro.float64:       cg_numba_cast(types.float64),
    numbapro.complex64:     cg_numba_cast(types.complex64),
    numbapro.complex128:    cg_numba_cast(types.complex128),
}

cudapy_call_codegen_ext.update(numba_cast_ext)
cudapy_call_codegen_ext.update(libdevice.math_codegen)
