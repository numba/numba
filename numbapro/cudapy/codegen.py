from llvm.core import Type

from numbapro.npm.types import *
from numbapro.npm.errors import CompileError
from . import ptx

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

#-------------------------------------------------------------------------------

cudapy_global_codegen_ext = {
    ptx.threadIdx.x: cg_sreg,
    ptx.threadIdx.y: cg_sreg,
    ptx.threadIdx.z: cg_sreg,
    ptx.blockIdx.x: cg_sreg,
    ptx.blockIdx.y: cg_sreg,

    ptx.blockDim.x: cg_sreg,
    ptx.blockDim.y: cg_sreg,
    ptx.blockDim.z: cg_sreg,
    ptx.gridDim.x: cg_sreg,
    ptx.gridDim.y: cg_sreg,
}

cudapy_call_codegen_ext = {
    ptx.grid: cg_grid_macro,
}
