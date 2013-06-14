from llvm.core import Type

from numbapro.npm.types import *
from . import ptx

def cg_threadIdx_x(cg, value):
    ty = cg.typemap[value]
    fname = ptx.SREG_MAPPING[value.args.value]
    fnty = Type.function(Type.int(), [])
    func = cg.lmod.get_or_insert_function(fnty, name=fname)
    res = cg.builder.call(func, ())
    return cg.do_cast(res, int32, ty)


#-------------------------------------------------------------------------------

cudapy_codegen_ext = {
    ptx.threadIdx.x: cg_threadIdx_x,
}
