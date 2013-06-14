from llvm.core import Type

from numbapro.npm.types import *
from . import ptx

def cg_sreg(cg, value):
    ty = cg.typemap[value]
    fname = ptx.SREG_MAPPING[value.args.value]
    fnty = Type.function(Type.int(), [])
    func = cg.lmod.get_or_insert_function(fnty, name=fname)
    res = cg.builder.call(func, ())
    return cg.do_cast(res, uint32, ty)

#-------------------------------------------------------------------------------

cudapy_codegen_ext = {
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
