from llvm.core import Type

from numbapro import cuda
from numbapro.npm.types import *
from numbapro.cudapipeline.special_values import sreg

def cg_threadIdx_x(cg, value):
    ty = cg.typemap[value]
    fname = sreg.SPECIAL_VALUES[value.args.value]
    fnty = Type.function(Type.int(), [])
    func = cg.lmod.get_or_insert_function(fnty, name=fname)
    res = cg.builder.call(func, ())
    return cg.do_cast(res, int32, ty)


#-------------------------------------------------------------------------------

cudapy_codegen_ext = {
    cuda.threadIdx.x: cg_threadIdx_x,
}
