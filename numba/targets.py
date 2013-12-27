from llvm.core import Type, Constant
import llvm.core as lc
import llvm.passes as lp
from numba import types, utils
from numba.typing import signature

LTYPEMAP = {
    types.boolean: Type.int(1),

    types.int32: Type.int(32),
    types.int64: Type.int(64),
}


class BaseContext(object):
    def __init__(self):
        self.defns = utils.UniqueDict()
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

    def get_function(self, fn, sig):
        key = fn, sig
        return self.defns[key]

    def get_return_value(self, builder, ty, val):
        if ty is types.boolean:
            r = self.get_return_type(ty)
            return builder.zext(val, r)
        else:
            return val

    def cast(self, builder, val, fromty, toty):
        if fromty == toty:
            return val
        else:
            raise NotImplementedError("cast", val, fromty, toty)

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

def implement(func, return_type, *args):
    def wrapper(impl):
        impl.signature = signature(return_type, *args)
        impl.key = func, impl.signature
        return impl
    return wrapper

BUILTINS = []


def builtin(impl):
    BUILTINS.append(impl)

#-------------------------------------------------------------------------------


@builtin
@implement('<', types.boolean, types.int32, types.int32)
def int_lt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SLT, *args)


@builtin
@implement('>', types.boolean, types.int32, types.int32)
def int_gt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SGT, *args)

