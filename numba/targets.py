from __future__ import print_function
from llvm.core import Type, Constant
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from numba import types, utils, cgutils, _dynfunc
from numba.typing import signature
from numba.callwrapper import PyCallWrapper
from numba.pythonapi import PythonAPI

LTYPEMAP = {
    types.pyobject: Type.pointer(Type.int(8)),

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
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """
        pass

    def get_argument_type(self, ty):
        if ty is types.boolean:
            return Type.int(8)
        else:
            return self.get_value_type(ty)

    def get_return_type(self, ty):
        return self.get_argument_type(ty)

    def get_value_type(self, ty):
        if isinstance(ty, types.Dummy):
            return self.get_dummy_type()
        elif ty == types.range_state32_type:
            stty = self.get_struct_type(RangeState32)
            return Type.pointer(stty)
        elif ty == types.range_iter32_type:
            stty = self.get_struct_type(RangeIter32)
            return Type.pointer(stty)
        return LTYPEMAP[ty]

    def get_constant(self, ty, val):
        lty = self.get_value_type(ty)
        if ty in types.signed_domain:
            return Constant.int_signextend(lty, val)

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

    def get_struct_type(self, struct):
        fields = [self.get_value_type(v) for _, v in struct._fields]
        return Type.struct(fields)

    def get_dummy_value(self):
        return Constant.null(self.get_dummy_type())

    def get_dummy_type(self):
        return Type.int()

    def optimize(self, module):
        pass

    def get_executable(self, func, fndesc):
        raise NotImplementedError

    def get_python_api(self, builder):
        return PythonAPI(self, builder)


class CPUContext(BaseContext):
    def init(self):
        self.execmodule = lc.Module.new("numba.exec")
        eb = le.EngineBuilder.new(self.execmodule).opt(3)
        self.tm = tm = eb.select_target()
        self.engine = eb.create(tm)

        pms = lp.build_pass_managers(tm=self.tm, loop_vectorize=True, opt=2,
                                     fpm=False)
        self.pm = pms.pm

        # self.pm = lp.PassManager.new()
        # self.pm.add(lp.Pass.new("mem2reg"))
        # self.pm.add(lp.Pass.new("simplifycfg"))

    def optimize(self, module):
        self.pm.run(module)

    def get_executable(self, func, fndesc):
        wrapper = PyCallWrapper(self, func.module, func, fndesc).build()
        self.optimize(func.module)
        print(func.module)
        self.engine.add_module(func.module)
        fnptr = self.engine.get_pointer_to_function(wrapper)

        func = _dynfunc.make_function(fnptr).dyncallable
        return func

#-------------------------------------------------------------------------------

def implement(func, return_type, *args):
    def wrapper(impl):
        def res(context, builder, args):
            return impl(context, builder, args)
        res.signature = signature(return_type, *args)
        res.key = func, res.signature
        return res
    return wrapper

BUILTINS = []


def builtin(impl):
    BUILTINS.append(impl)
    return impl

#-------------------------------------------------------------------------------

def int_add_impl(context, builder, args):
    return builder.add(*args)


def int_sub_impl(context, builder, args):
    return builder.sub(*args)


def int_mul_impl(context, builder, args):
    return builder.mul(*args)


def int_udiv_impl(context, builder, args):
    return builder.udiv(*args)


def int_sdiv_impl(context, builder, args):
    return builder.sdiv(*args)


def int_slt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SLT, *args)


def int_sle_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SLE, *args)


def int_sgt_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SGT, *args)


def int_sge_impl(context, builder, args):
    return builder.icmp(lc.ICMP_SGE, *args)


def int_eq_impl(context, builder, args):
    return builder.icmp(lc.ICMP_EQ, *args)


def int_ne_impl(context, builder, args):
    return builder.icmp(lc.ICMP_NE, *args)


for ty in types.integer_domain:
    builtin(implement('+', ty, ty, ty)(int_add_impl))
    builtin(implement('-', ty, ty, ty)(int_sub_impl))
    builtin(implement('*', ty, ty, ty)(int_mul_impl))
    builtin(implement('==', types.boolean, ty, ty)(int_eq_impl))
    builtin(implement('!=', types.boolean, ty, ty)(int_ne_impl))

for ty in types.unsigned_domain:
    builtin(implement('/?', ty, ty, ty)(int_udiv_impl))


for ty in types.signed_domain:
    builtin(implement('/?', ty, ty, ty)(int_sdiv_impl))
    builtin(implement('<', types.boolean, ty, ty)(int_slt_impl))
    builtin(implement('<=', types.boolean, ty, ty)(int_sle_impl))
    builtin(implement('>', types.boolean, ty, ty)(int_sgt_impl))
    builtin(implement('>=', types.boolean, ty, ty)(int_sge_impl))


class RangeState32(cgutils.Structure):
    _fields = [('start',  types.int32),
               ('stop',  types.int32),
               ('step',  types.int32)]


class RangeIter32(cgutils.Structure):
    _fields = [('iter',  types.int32),
               ('stop',  types.int32),
               ('step',  types.int32),
               ('count', types.int32)]


@builtin
@implement(types.range_type, types.range_state32_type, types.int32, types.int32)
def range2_32_impl(context, builder, args):
    start, stop = args
    state = RangeState32(context, builder)

    state.start = start
    state.stop = stop
    state.step = context.get_constant(types.int32, 1)

    return state._getvalue()


@builtin
@implement('getiter', types.range_iter32_type, types.range_state32_type)
def getiter_range32_impl(context, builder, args):
    (value,) = args
    state = RangeState32(context, builder, value)
    iterobj = RangeIter32(context, builder)

    start = state.start
    stop = state.stop
    step = state.step

    iterobj.iter = start
    iterobj.stop = stop
    iterobj.step = step
    iterobj.count = builder.sdiv(builder.sub(stop, start), step)

    return iterobj._getvalue()


@builtin
@implement('iternext', types.int32, types.range_iter32_type)
def iternext_range32_impl(context, builder, args):
    (value,) = args
    iterobj = RangeIter32(context, builder, value)

    res = iterobj.iter
    one = context.get_constant(types.int32, 1)
    iterobj.count = builder.sub(iterobj.count, one)
    iterobj.iter = builder.add(res, iterobj.step)

    return res


@builtin
@implement('itervalid', types.boolean, types.range_iter32_type)
def itervalid_range32_impl(context, builder, args):
    (value,) = args
    iterobj = RangeIter32(context, builder, value)

    zero = context.get_constant(types.int32, 0)
    gt = builder.icmp(lc.ICMP_SGE, iterobj.count, zero)
    return gt
