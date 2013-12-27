from llvm.core import Type, Constant
import llvm.core as lc
import llvm.passes as lp
from numba import types
from numba.typing import signature

LTYPEMAP = {
    types.boolean: Type.int(1),

    types.int32: Type.int(32),
    types.int64: Type.int(64),
}


class BaseContext(object):
    def __init__(self):
        self.defns = {}
        # Initialize
        for defn in BUILTINS:
            self.defns[defn.key] = defn

    def get_argument_type(self, ty):
        if ty is types.boolean:
            return Type.int(8)
        else:
            return self.get_value_type(ty)

    def get_return_type(self, ty):
        return self.get_argument_type(ty)

    def get_value_type(self, ty):
        return LTYPEMAP[ty]

    def get_constant(self, ty, val):
        lty = self.get_value_type(ty)
        if ty in types.signed_domain:
            return Constant.int(lty, val)

    def get_function(self, fn, *types):
        defn = self.defns[fn]
        for impl in defn.cases:
            sig = impl.signature
            if sig.args == types:
                return impl
        else:
            raise NotImplementedError(fn, types)

    def get_return_value(self, builder, ty, val):
        if ty is types.boolean:
            r = self.get_return_type(ty)
            return builder.zext(val, r)
        else:
            return val

    def optimize(self, module):
        pass


class CPUContext(BaseContext):
    def optimize(self, module):
        pmb = lp.PassManagerBuilder.new()
        pmb.opt_level = 2
        pm = lp.PassManager.new()
        pmb.populate(pm)
        pm.run(module)


#-------------------------------------------------------------------------------

def implement(return_type, *args):
    def wrapper(fn):
        fn.signature = signature(return_type, *args)
        return fn

    return wrapper


@implement(types.boolean, types.int32, types.int32)
def int_lt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SLT, *args)


@implement(types.boolean, types.int32, types.int32)
def int_gt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SGT, *args)


class CmpOpLt(object):
    key = '<'
    cases = [
        int_lt_impl,
    ]


class CmpOpGt(object):
    key = '>'
    cases = [
        int_gt_impl,
    ]


BUILTINS = [
    CmpOpLt,
    CmpOpGt,
]