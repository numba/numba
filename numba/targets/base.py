from __future__ import print_function
import math
from collections import namedtuple, defaultdict
import functools
import llvm.core as lc
from llvm.core import Type, Constant

from numba import types, utils, cgutils, typing
from numba.typing import signature
from numba.pythonapi import PythonAPI


LTYPEMAP = {
    types.pyobject: Type.pointer(Type.int(8)),

    types.boolean: Type.int(1),

    types.uint8: Type.int(8),
    types.int32: Type.int(32),
    types.int64: Type.int(64),

    types.float32: Type.float(),
    types.float64: Type.double(),
}


Status = namedtuple("Status", ("code", "ok", "err"))


class Overloads(object):
    def __init__(self):
        self.versions = []

    def find(self, sig):
        for ver in self.versions:
            if ver.signature == sig:
                return ver
            # As generic type
            if len(ver.signature.args) == len(sig.args):
                match = True
                for formal, actual in zip(ver.signature.args, sig.args):
                    if formal == actual:
                        pass
                    elif types.any == formal:
                        pass
                    elif (isinstance(formal, types.Kind) and
                          isinstance(actual, formal.of)):
                        pass
                    else:
                        match = False
                        break

                if match:
                    return ver

        raise NotImplementedError(self, sig)

    def append(self, impl):
        self.versions.append(impl)


class BaseContext(object):
    def __init__(self):
        self.defns = defaultdict(Overloads)
        self.attrs = utils.UniqueDict()
        self.users = utils.UniqueDict()

        for defn in BUILTINS:
            self.defns[defn.key].append(defn)
        for attr in BUILTIN_ATTRS:
            self.attrs[attr.key] = attr

        # Initialize
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """
        pass

    def insert_user_function(self, func, fndesc):
        imp = user_function(func, fndesc)
        self.defns[func].append(imp)

        class UserFunction(typing.templates.ConcreteTemplate):
            key = func
            cases = [imp.signature]

        self.users[func] = UserFunction

    def get_user_function(self, func):
        return self.users[func]

    def get_function_type(self, fndesc):
        """
        Calling Convention
        ------------------
        Returns: -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.
        The 2nd argument is a reference to the owner python module.
        It is used to get global values inside the function.
        It MUST NOT be used if the function is in nopython mode.
        Actual arguments starts at the 3nd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        restype = self.get_return_type(fndesc.restype)
        scope = self.get_argument_type(types.pyobject)
        resptr = Type.pointer(restype)
        fnty = Type.function(Type.int(), [resptr, scope] + argtypes)
        return fnty

    def declare_function(self, module, fndesc):
        fnty = self.get_function_type(fndesc)
        fn = module.get_or_insert_function(fnty, name=fndesc.name)
        assert fn.is_declaration
        for ak, av in zip(fndesc.args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        fn.args[0] = ".ret"
        return fn

    def insert_const_string(self, mod, string):
        stringtype = Type.pointer(Type.int(8))
        text = Constant.stringz(string)
        name = ".const.%s" % string
        gv = mod.add_global_variable(text.type, name=name)
        gv.global_constant = True
        gv.initializer = text
        gv.linkage = lc.LINKAGE_INTERNAL
        return Constant.bitcast(gv, stringtype)

    def get_scope(self, func):
        return func.args[1]

    def get_arguments(self, func):
        return func.args[2:]

    def get_argument_type(self, ty):
        if ty is types.boolean:
            return Type.int(8)
        else:
            return self.get_value_type(ty)

    def get_return_type(self, ty):
        return self.get_argument_type(ty)

    def get_value_type(self, ty):
        if (isinstance(ty, types.Dummy) or
                isinstance(ty, types.Module) or
                isinstance(ty, types.Function)):
            return self.get_dummy_type()
        elif ty == types.range_state32_type:
            stty = self.get_struct_type(RangeState32)
            return Type.pointer(stty)
        elif ty == types.range_iter32_type:
            stty = self.get_struct_type(RangeIter32)
            return Type.pointer(stty)
        elif ty == types.range_state64_type:
            stty = self.get_struct_type(RangeState64)
            return Type.pointer(stty)
        elif ty == types.range_iter64_type:
            stty = self.get_struct_type(RangeIter64)
            return Type.pointer(stty)
        elif isinstance(ty, types.Array):
            stty = self.get_struct_type(make_array(ty))
            return Type.pointer(stty)
        elif isinstance(ty, types.CPointer):
            dty = self.get_value_type(ty.dtype)
            return Type.pointer(dty)
        elif isinstance(ty, types.UniTuple):
            dty = self.get_value_type(ty.dtype)
            return Type.array(dty, ty.count)
        return LTYPEMAP[ty]

    def get_constant(self, ty, val):
        lty = self.get_value_type(ty)

        if ty == types.none:
            assert val is None
            return self.get_dummy_value()

        elif ty in types.signed_domain:
            return Constant.int_signextend(lty, val)

        elif ty in types.real_domain:
            return Constant.real(lty, val)

        raise NotImplementedError(ty)

    def get_constant_undef(self, ty):
        lty = self.get_value_type(ty)
        return Constant.undef(lty)

    def get_constant_null(self, ty):
        lty = self.get_value_type(ty)
        return Constant.null(lty)

    def get_function(self, fn, sig):
        if isinstance(fn, types.Function):
            overloads = self.defns[fn.template.key]
        else:
            overloads = self.defns[fn]
        try:
            return overloads.find(sig)
        except NotImplementedError:
            raise NotImplementedError(fn, sig)

    def get_attribute(self, val, typ, attr):
        key = typ, attr
        try:
            return self.attrs[key]
        except KeyError:
            if isinstance(typ, types.Module):
                return
            elif typ.is_parametric:
                key = type(typ), attr
                return self.attrs[key]
            else:
                raise

    def get_return_value(self, builder, ty, val):
        if ty is types.boolean:
            r = self.get_return_type(ty)
            return builder.zext(val, r)
        else:
            return val

    def return_value(self, builder, retval):
        fn = builder.basic_block.function
        retptr = fn.args[0]
        assert retval.type == retptr.type.pointee
        builder.store(retval, retptr)
        builder.ret(Constant.null(Type.int()))

    def return_errcode(self, builder, code):
        assert code > 0
        builder.ret(Constant.int(Type.int(), code))

    def return_exc(self, builder):
        builder.ret(Constant.int_signextend(Type.int(), -1))

    def cast(self, builder, val, fromty, toty):
        if fromty == toty or toty == types.any or isinstance(toty, types.Kind):
            return val

        elif fromty in types.signed_domain and toty in types.signed_domain:
            lfrom = self.get_value_type(fromty)
            lto = self.get_value_type(toty)
            if lfrom.width < lto.width:
                return builder.sext(val, lto)
            elif lfrom.width > lto.width:
                return builder.zext(val, lto)

        elif fromty in types.real_domain and toty in types.real_domain:
            lty = self.get_value_type(toty)
            if fromty == types.float32 and toty == types.float64:
                return builder.fpext(val, lty)
            elif fromty == types.float64 and toty == types.float32:
                return builder.fptrunc(val, lty)

        elif fromty in types.integer_domain and toty in types.real_domain:
            lty = self.get_value_type(toty)
            if fromty in types.signed_domain:
                return builder.sitofp(val, lty)
            else:
                return builder.uitofp(val, lty)

        elif toty in types.integer_domain and fromty in types.real_domain:
            lty = self.get_value_type(toty)
            if toty in types.signed_domain:
                return builder.fptosi(val, lty)
            else:
                return builder.fptoui(val, lty)


        raise NotImplementedError("cast", val, fromty, toty)

    def call_function(self, builder, scope, callee, args):
        retty = callee.args[0].type.pointee
        # TODO: user supplied retval or let user tell where to allocate
        retval = builder.alloca(retty)
        realargs = [retval, scope] + list(args)
        code = builder.call(callee, realargs)
        ok = builder.icmp(lc.ICMP_EQ, code, Constant.null(Type.int()))
        err = builder.not_(ok)

        status = Status(code=code, ok=ok, err=err)
        return status, builder.load(retval)

    def call_function_native(self, builder, callee, args):
        """
        Call a native function generated by this context.
        The owner-pymodule (2nd argument) is NULL.
        """
        scope = self.get_constant_null(types.pyobject)
        self.call_function(builder, scope, callee, args)

    def print_string(self, builder, text):
        mod = builder.basic_block.function.module
        cstring = Type.pointer(Type.int(8))
        fnty = Type.function(Type.int(), [cstring])
        puts = mod.get_or_insert_function(fnty, "puts")
        return builder.call(puts, [text])

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

    def get_python_api(self, builder, ownmod=None):
        return PythonAPI(self, builder, ownmod)

    def make_array(self, typ):
        return make_array(typ)


#-------------------------------------------------------------------------------


def make_array(ty):
    dtype = ty.dtype
    nd = ty.ndim

    class ArrayTemplate(cgutils.Structure):
        _fields = [('data',    types.CPointer(dtype)),
                   ('shape',   types.UniTuple(types.intp, nd)),
                   ('strides', types.UniTuple(types.intp, nd)),]

    return ArrayTemplate

#-------------------------------------------------------------------------------


def implement(func, return_type, *args):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, tys, args):
            ret = impl(context, builder, tys, args)
            return ret
        res.signature = signature(return_type, *args)
        res.key = func
        return res
    return wrapper


def impl_attribute(ty, attr, rtype):
    def wrapper(impl):
        @functools.wraps(impl)
        def res(context, builder, typ, value):
            ret = impl(context, builder, typ, value)
            return ret
        res.return_type = rtype
        res.key = (ty, attr)
        return res
    return wrapper


def user_function(func, fndesc):
    @implement(func, fndesc.restype, *fndesc.argtypes)
    def imp(context, builder, tys, args):
        func = context.declare_function(cgutils.get_module(builder), fndesc)
        scope = Constant.null(Type.pointer(Type.int(8)))
        status, retval = context.call_function(builder, scope, func, args)
        # TODO handling error
        return retval
    return imp

#-------------------------------------------------------------------------------


BUILTINS = []
BUILTIN_ATTRS = []


def builtin(impl):
    BUILTINS.append(impl)
    return impl


def builtin_attr(impl):
    BUILTIN_ATTRS.append(impl)
    return impl

#-------------------------------------------------------------------------------


def int_add_impl(context, builder, tys, args):
    return builder.add(*args)


def int_sub_impl(context, builder, tys, args):
    return builder.sub(*args)


def int_mul_impl(context, builder, tys, args):
    return builder.mul(*args)


def int_udiv_impl(context, builder, tys, args):
    return builder.udiv(*args)


def int_sdiv_impl(context, builder, tys, args):
    return builder.sdiv(*args)


def int_slt_impl(context, builder, tys, args):
    return builder.icmp(lc.ICMP_SLT, *args)


def int_sle_impl(context, builder, tys, args):
    return builder.icmp(lc.ICMP_SLE, *args)


def int_sgt_impl(context, builder, tys, args):
    return builder.icmp(lc.ICMP_SGT, *args)


def int_sge_impl(context, builder, tys, args):
    return builder.icmp(lc.ICMP_SGE, *args)


def int_eq_impl(context, builder, tys, args):
    return builder.icmp(lc.ICMP_EQ, *args)


def int_ne_impl(context, builder, tys, args):
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



def real_add_impl(context, builder, tys, args):
    return builder.fadd(*args)


def real_sub_impl(context, builder, tys, args):
    return builder.fsub(*args)


def real_mul_impl(context, builder, tys, args):
    return builder.fmul(*args)


def real_div_impl(context, builder, tys, args):
    return builder.fdiv(*args)


def real_lt_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OLT, *args)


def real_le_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OLE, *args)


def real_gt_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OGT, *args)


def real_ge_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OGE, *args)


def real_eq_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_UEQ, *args)


def real_ne_impl(context, builder, tys, args):
    return builder.icmp(lc.FCMP_ONE, *args)


for ty in types.real_domain:
    builtin(implement('+', ty, ty, ty)(real_add_impl))
    builtin(implement('-', ty, ty, ty)(real_sub_impl))
    builtin(implement('*', ty, ty, ty)(real_mul_impl))
    builtin(implement('/?', ty, ty, ty)(real_div_impl))
    builtin(implement('/', ty, ty, ty)(real_div_impl))
    builtin(implement('==', types.boolean, ty, ty)(real_eq_impl))
    builtin(implement('!=', types.boolean, ty, ty)(real_ne_impl))
    builtin(implement('<', types.boolean, ty, ty)(real_lt_impl))
    builtin(implement('<=', types.boolean, ty, ty)(real_le_impl))
    builtin(implement('>', types.boolean, ty, ty)(real_gt_impl))
    builtin(implement('>=', types.boolean, ty, ty)(real_ge_impl))


class RangeState32(cgutils.Structure):
    _fields = [('start', types.int32),
               ('stop',  types.int32),
               ('step',  types.int32)]


class RangeIter32(cgutils.Structure):
    _fields = [('iter',  types.int32),
               ('stop',  types.int32),
               ('step',  types.int32),
               ('count', types.int32)]


class RangeState64(cgutils.Structure):
    _fields = [('start', types.int64),
               ('stop',  types.int64),
               ('step',  types.int64)]


class RangeIter64(cgutils.Structure):
    _fields = [('iter',  types.int64),
               ('stop',  types.int64),
               ('step',  types.int64),
               ('count', types.int64)]


@builtin
@implement(types.range_type, types.range_state32_type, types.int32, types.int32)
def range2_32_impl(context, builder, tys, args):
    start, stop = args
    state = RangeState32(context, builder)

    state.start = start
    state.stop = stop
    state.step = context.get_constant(types.int32, 1)

    return state._getvalue()


@builtin
@implement('getiter', types.range_iter32_type, types.range_state32_type)
def getiter_range32_impl(context, builder, tys, args):
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
def iternext_range32_impl(context, builder, tys, args):
    (value,) = args
    iterobj = RangeIter32(context, builder, value)

    res = iterobj.iter
    one = context.get_constant(types.int32, 1)
    iterobj.count = builder.sub(iterobj.count, one)
    iterobj.iter = builder.add(res, iterobj.step)

    return res


@builtin
@implement('itervalid', types.boolean, types.range_iter32_type)
def itervalid_range32_impl(context, builder, tys, args):
    (value,) = args
    iterobj = RangeIter32(context, builder, value)

    zero = context.get_constant(types.int32, 0)
    gt = builder.icmp(lc.ICMP_SGE, iterobj.count, zero)
    return gt



@builtin
@implement(types.range_type, types.range_state64_type, types.int64)
def range1_64_impl(context, builder, tys, args):
    (stop,) = args
    state = RangeState64(context, builder)

    state.start = context.get_constant(types.int64, 0)
    state.stop = stop
    state.step = context.get_constant(types.int64, 1)

    return state._getvalue()


@builtin
@implement(types.range_type, types.range_state64_type, types.int64,
           types.int64)
def range2_64_impl(context, builder, tys, args):
    start, stop = args
    state = RangeState64(context, builder)

    state.start = start
    state.stop = stop
    state.step = context.get_constant(types.int64, 1)

    return state._getvalue()


@builtin
@implement('getiter', types.range_iter64_type, types.range_state64_type)
def getiter_range64_impl(context, builder, tys, args):
    (value,) = args
    state = RangeState64(context, builder, value)
    iterobj = RangeIter64(context, builder)

    start = state.start
    stop = state.stop
    step = state.step

    iterobj.iter = start
    iterobj.stop = stop
    iterobj.step = step
    iterobj.count = builder.sdiv(builder.sub(stop, start), step)

    return iterobj._getvalue()


@builtin
@implement('iternext', types.int64, types.range_iter64_type)
def iternext_range64_impl(context, builder, tys, args):
    (value,) = args
    iterobj = RangeIter64(context, builder, value)

    res = iterobj.iter
    one = context.get_constant(types.int64, 1)
    iterobj.count = builder.sub(iterobj.count, one)
    iterobj.iter = builder.add(res, iterobj.step)

    return res


@builtin
@implement('itervalid', types.boolean, types.range_iter64_type)
def itervalid_range64_impl(context, builder, tys, args):
    (value,) = args
    iterobj = RangeIter64(context, builder, value)

    zero = context.get_constant(types.int64, 0)
    gt = builder.icmp(lc.ICMP_SGE, iterobj.count, zero)
    return gt


@builtin
@implement('getitem', types.any, types.Kind(types.UniTuple), types.intp)
def getitem_unituple(context, builder, tys, args):
    tupty, _ = tys
    tup, idx = args

    bbelse = cgutils.append_basic_block(builder, "switch.else")
    bbend = cgutils.append_basic_block(builder, "switch.end")
    switch = builder.switch(idx, bbelse, n=tupty.count)

    with cgutils.goto_block(builder, bbelse):
        # TODO: propagate exception to
        context.return_errcode(builder, 1)

    lrtty = context.get_value_type(tupty.dtype)
    with cgutils.goto_block(builder, bbend):
        phinode = builder.phi(lrtty)

    for i in range(tupty.count):
        ki = context.get_constant(types.intp, i)
        bbi = cgutils.append_basic_block(builder, "switch.%d" % i)
        switch.add_case(ki, bbi)
        with cgutils.goto_block(builder, bbi):
            value = builder.extract_value(tup, i)
            builder.branch(bbend)
            phinode.add_incoming(value, bbi)

    builder.position_at_end(bbend)
    return phinode


@builtin
@implement('getitem', types.any, types.Kind(types.Array), types.intp)
def getitem_array1d(context, builder, tys, args):
    aryty, _ = tys
    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    dataptr = ary.data

    ptr = builder.gep(dataptr, [idx])
    return builder.load(ptr)


@builtin
@implement('getitem', types.any, types.Kind(types.Array),
           types.Kind(types.UniTuple))
def getitem_array_unituple(context, builder, tys, args):
    aryty, idxty = tys
    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other than layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices)
    return builder.load(ptr)


@builtin
@implement('setitem', types.none, types.Kind(types.Array), types.intp,
           types.any)
def setitem_array1d(context, builder, tys, args):
    aryty, _, valty = tys
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    dataptr = ary.data

    ptr = builder.gep(dataptr, [idx])
    builder.store(val, ptr)
    return


@builtin
@implement('setitem', types.none, types.Kind(types.Array),
           types.Kind(types.UniTuple), types.any)
def setitem_array_unituple(context, builder, tys, args):
    aryty, idxty, valty = tys
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other than layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices)
    builder.store(val, ptr)


@builtin
@implement(types.len_type, types.intp, types.Kind(types.Array))
def array_len(context, builder, tys, args):
    (aryty,) = tys
    (ary,) = args
    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    shapeary = ary.shape
    return builder.extract_value(shapeary, 0)

#-------------------------------------------------------------------------------


@builtin_attr
@impl_attribute(types.Array, "shape", types.UniTuple)
def array_shape(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.shape

#-------------------------------------------------------------------------------


@builtin
@implement(math.fabs, types.float32, types.float32)
def math_fabs_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_FABS, [lty])
    return builder.call(intr, args)



@builtin
@implement(math.fabs, types.float64, types.float64)
def math_fabs_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_FABS, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.exp, types.float32, types.float32)
def math_exp_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_EXP, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.exp, types.float64, types.float64)
def math_exp_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_EXP, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.sqrt, types.float32, types.float32)
def math_sqrt_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_SQRT, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.sqrt, types.float64, types.float64)
def math_sqrt_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_SQRT, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.log, types.float32, types.float32)
def math_log_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_LOG, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.log, types.float64, types.float64)
def math_log_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_LOG, [lty])
    return builder.call(intr, args)