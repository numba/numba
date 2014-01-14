from __future__ import print_function
import math
from collections import namedtuple, defaultdict
import functools
import llvm.core as lc
from llvm.core import Type, Constant
import numpy

from numba import types, utils, cgutils, typing
from numba.typing import signature
from numba.pythonapi import PythonAPI


LTYPEMAP = {
    types.pyobject: Type.pointer(Type.int(8)),

    types.boolean: Type.int(1),

    types.uint8: Type.int(8),
    types.uint16: Type.int(16),
    types.uint32: Type.int(32),
    types.uint64: Type.int(64),

    types.int8: Type.int(8),
    types.int16: Type.int(16),
    types.int32: Type.int(32),
    types.int64: Type.int(64),

    types.float32: Type.float(),
    types.float64: Type.double(),
}


Status = namedtuple("Status", ("code", "ok", "err", "none"))


RETCODE_OK = Constant.int_signextend(Type.int(), 0)
RETCODE_NONE = Constant.int_signextend(Type.int(), -2)
RETCODE_EXC = Constant.int_signextend(Type.int(), -1)


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
                    match = self._match(formal, actual)
                    if not match:
                        break

                if match:
                    return ver

        raise NotImplementedError(self, sig)

    @staticmethod
    def _match(formal, actual):
        if formal == actual:
            # formal argument matches actual arguments
            return True
        elif types.Any == formal:
            # formal argument is any
            return True
        elif (isinstance(formal, types.Kind) and
              isinstance(actual, formal.of)):
            # formal argument is a kind and the actual argument
            # is of that kind
            return True

    def append(self, impl):
        self.versions.append(impl)


class BaseContext(object):
    """

    Notes on Structure
    ------------------

    Most objects are lowered as plain-old-data structure in the generated
    llvm.  They are passed around by reference (a pointer to the structure).
    Only POD structure can life across function boundaries by copying the
    data.
    """
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

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        for name, vtype in attrs.iteritems():
            imp = python_attr_impl(clsty, name, vtype)
            self.attrs[imp.key] = imp

    def get_user_function(self, func):
        return self.users[func]

    def get_function_type(self, fndesc):
        """
        Calling Convention
        ------------------
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.
        It MUST NOT be used if the function is in nopython mode.
        Actual arguments starts at the 2nd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        restype = self.get_return_type(fndesc.restype)
        resptr = Type.pointer(restype)
        fnty = Type.function(Type.int(), [resptr] + argtypes)
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
        for gv in mod.global_variables:
            if gv.name == name and gv.type.pointee == text.type:
                break
        else:
            gv = mod.add_global_variable(text.type, name=name)
            gv.global_constant = True
            gv.initializer = text
            gv.linkage = lc.LINKAGE_INTERNAL
        return Constant.bitcast(gv, stringtype)

    def get_arguments(self, func):
        return func.args[1:]

    def get_argument_type(self, ty):
        if ty is types.boolean:
            return Type.int(8)
        else:
            return self.get_value_type(ty)

    def get_return_type(self, ty):
        if self.is_struct_type(ty):
            vty = self.get_value_type(ty)
            return vty.pointee
        else:
            return self.get_argument_type(ty)

    def get_value_type(self, ty):
        if (isinstance(ty, types.Dummy) or
                isinstance(ty, types.Module) or
                isinstance(ty, types.Function) or
                isinstance(ty, types.Object)):
            return self.get_dummy_type()
        elif isinstance(ty, types.Optional):
            return self.get_value_type(ty.type)
        elif ty == types.complex64:
            stty = self.get_struct_type(Complex64)
            return Type.pointer(stty)
        elif ty == types.complex128:
            stty = self.get_struct_type(Complex128)
            return Type.pointer(stty)
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
        elif ty == types.slice3_type:
            stty = self.get_struct_type(Slice)
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
        elif isinstance(ty, types.UniTupleIter):
            stty = self.get_struct_type(self.make_unituple_iter(ty))
            return Type.pointer(stty)
        return LTYPEMAP[ty]

    def is_struct_type(self, ty):
        if isinstance(ty, types.Array):
            return True

        sttys = [
            types.complex64, types.complex128,
            types.range_state32_type, types.range_state64_type,
            types.range_iter32_type, types.range_iter64_type,
            types.slice2_type, types.slice3_type,
        ]
        return ty in sttys

    def get_constant_struct(self, builder, ty, val):
        assert self.is_struct_type(ty)
        module = cgutils.get_module(builder)

        if ty in types.complex_domain:
            if ty == types.complex64:
                innertype = types.float32
            elif ty == types.complex128:
                innertype = types.float64
            else:
                raise Exception("unreachable")

            real = self.get_constant(innertype, val.real)
            imag = self.get_constant(innertype, val.imag)
            const = Constant.struct([real, imag])

            gv = module.add_global_variable(const.type, name=".const")
            gv.linkage = lc.LINKAGE_INTERNAL
            gv.initializer = const
            gv.global_constant = True
            return gv

        else:
            raise NotImplementedError(ty)

    def get_constant(self, ty, val):
        assert not self.is_struct_type(ty)

        lty = self.get_value_type(ty)

        if ty == types.none:
            assert val is None
            return self.get_dummy_value()

        elif ty == types.boolean:
            return Constant.int(Type.int(1), int(val))

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
        if isinstance(fn, types.Method):
            return self.call_method
        elif isinstance(fn, types.Function):
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
        fn = cgutils.get_function(builder)
        retptr = fn.args[0]
        if (retval.type.kind == lc.TYPE_POINTER and
                retval.type.pointee.kind == lc.TYPE_STRUCT):
            # Copy structure
            builder.store(builder.load(retval), retptr)
        else:
            assert retval.type == retptr.type.pointee, \
                        (str(retval.type), str(retptr.type.pointee))
            builder.store(retval, retptr)
        builder.ret(RETCODE_OK)

    def return_native_none(self, builder):
        builder.ret(RETCODE_NONE)

    def return_errcode(self, builder, code):
        assert code > 0
        builder.ret(Constant.int(Type.int(), code))

    def return_exc(self, builder):
        builder.ret(RETCODE_EXC)

    def cast(self, builder, val, fromty, toty):
        if fromty == toty or toty == types.Any or isinstance(toty, types.Kind):
            return val

        elif fromty in types.unsigned_domain and toty in types.signed_domain:
            lfrom = self.get_value_type(fromty)
            lto = self.get_value_type(toty)
            if lfrom.width <= lto.width:
                return builder.zext(val, lto)
            elif lfrom.width > lto.width:
                return builder.trunc(val, lto)

        elif fromty in types.signed_domain and toty in types.signed_domain:
            lfrom = self.get_value_type(fromty)
            lto = self.get_value_type(toty)
            if lfrom.width <= lto.width:
                return builder.sext(val, lto)
            elif lfrom.width > lto.width:
                return builder.trunc(val, lto)

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

        elif fromty in types.integer_domain and toty in types.complex_domain:
            cmplxcls, flty = get_complex_info(toty)
            cmpl = cmplxcls(self, builder)
            cmpl.real = self.cast(builder, val, fromty, flty)
            cmpl.imag = self.get_constant(flty, 0)
            return cmpl._getvalue()

        elif fromty in types.complex_domain and toty in types.complex_domain:
            srccls, srcty = get_complex_info(fromty)
            dstcls, dstty = get_complex_info(toty)

            src = srccls(self, builder, value=val)
            dst = dstcls(self, builder)
            dst.real = self.cast(builder, src.real, srcty, dstty)
            dst.imag = self.cast(builder, src.imag, srcty, dstty)
            return dst._getvalue()

        elif (isinstance(toty, types.UniTuple) and
                  isinstance(fromty, types.UniTuple) and
                  len(fromty) == len(toty)):
            olditems = cgutils.unpack_tuple(builder, val, len(fromty))
            items = [self.cast(builder, i, fromty.dtype, toty.dtype)
                     for i in olditems]
            tup = self.get_constant_undef(toty)
            for idx, val in enumerate(items):
                tup = builder.insert_value(tup, val, idx)
            return tup

        raise NotImplementedError("cast", val, fromty, toty)

    def call_function(self, builder, callee, args):
        retty = callee.args[0].type.pointee
        retval = cgutils.alloca_once(builder, retty)
        realargs = [retval] + list(args)
        code = builder.call(callee, realargs)
        status = self.get_return_status(builder, code)

        if retval.type.pointee.kind == lc.TYPE_STRUCT:
            # Handle structures
            return status, retval
        else:
            return status, builder.load(retval)

    def get_return_status(self, builder, code):
        norm = builder.icmp(lc.ICMP_EQ, code, RETCODE_OK)
        none = builder.icmp(lc.ICMP_EQ, code, RETCODE_NONE)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)

        status = Status(code=code, ok=ok, err=err, none=none)
        return status

    def call_class_method(self, builder, func, retty, tys, args):
        api = self.get_python_api(builder)
        pyargs = [api.from_native_value(av, at) for av, at in zip(args, tys)]
        res = api.call_function_objargs(func, pyargs)

        # clean up
        api.decref(func)
        for obj in pyargs:
            api.decref(obj)

        with cgutils.ifthen(builder, cgutils.is_null(builder, res)):
            self.return_exc(builder)

        if retty == types.none:
            api.decref(res)
            return self.get_dummy_value()
        else:
            nativeresult = api.to_native_value(res, retty)
            api.decref(res)
            return nativeresult

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
        return Type.pointer(Type.int(8))

    def optimize(self, module):
        pass

    def get_executable(self, func, fndesc):
        raise NotImplementedError

    def get_python_api(self, builder):
        return PythonAPI(self, builder)

    def make_array(self, typ):
        return make_array(typ)

    def make_complex(self, typ):
        cls, _ = get_complex_info(typ)
        return cls

    def make_unituple_iter(self, typ):
        return make_unituple_iter(typ)
#-------------------------------------------------------------------------------


def make_array(ty):
    dtype = ty.dtype
    if dtype == types.boolean:
        dtype = types.byte

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
        status, retval = context.call_function(builder, func, args)
        # TODO handling error
        return retval
    return imp


def python_attr_impl(cls, attr, atyp):
    @impl_attribute(cls, attr, atyp)
    def imp(context, builder, typ, value):
        api = context.get_python_api(builder)
        aval = api.object_getattr_string(value, attr)
        with cgutils.ifthen(builder, cgutils.is_null(builder, aval)):
            context.return_exc(builder)

        if isinstance(atyp, types.Method):
            return aval
        else:
            nativevalue = api.to_native_value(aval, atyp)
            api.decref(aval)
            return nativevalue
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


def int_divmod(context, builder, x, y):
    """
    Reference Objects/intobject.c
    xdivy = x / y;
    xmody = (long)(x - (unsigned long)xdivy * y);
    /* If the signs of x and y differ, and the remainder is non-0,
     * C89 doesn't define whether xdivy is now the floor or the
     * ceiling of the infinitely precise quotient.  We want the floor,
     * and we have it iff the remainder's sign matches y's.
     */
    if (xmody && ((y ^ xmody) < 0) /* i.e. and signs differ */) {
        xmody += y;
        --xdivy;
        assert(xmody && ((y ^ xmody) >= 0));
    }
    *p_xdivy = xdivy;
    *p_xmody = xmody;
    """
    assert x.type == y.type
    xdivy = builder.sdiv(x, y)
    xmody = builder.srem(x, y)  # Intel has divmod instruction

    ZERO = Constant.null(y.type)
    ONE = Constant.int(y.type, 1)

    y_xor_xmody_ltz = builder.icmp(lc.ICMP_SLT, builder.xor(y, xmody), ZERO)
    xmody_istrue = builder.icmp(lc.ICMP_NE, xmody, ZERO)
    cond = builder.and_(xmody_istrue, y_xor_xmody_ltz)

    bb1 = builder.basic_block
    with cgutils.ifthen(builder, cond):
        xmody_plus_y = builder.add(xmody, y)
        xdivy_minus_1 = builder.sub(xdivy, ONE)
        bb2 = builder.basic_block

    resdiv = builder.phi(y.type)
    resdiv.add_incoming(xdivy, bb1)
    resdiv.add_incoming(xdivy_minus_1, bb2)

    resmod = builder.phi(x.type)
    resmod.add_incoming(xmody, bb1)
    resmod.add_incoming(xmody_plus_y, bb2)

    return resdiv, resmod


def int_sdiv_impl(context, builder, tys, args):
    x, y = args
    div, _ = int_divmod(context, builder, x, y)
    return div


def int_srem_impl(context, builder, tys, args):
    x, y = args
    _, rem = int_divmod(context, builder, x, y)
    return rem


def int_urem_impl(context, builder, tys, args):
    x, y = args
    return builder.urem(x, y)


def power_int_impl(context, builder, tys, args):
    module = cgutils.get_module(builder)
    x, y = args
    powerfn = lc.Function.intrinsic(module, lc.INTR_POWI, [x.type])
    return builder.call(powerfn, (x, y))


def int_power_func_body(context, builder, x, y):
    pcounter = builder.alloca(y.type)
    presult = builder.alloca(x.type)
    result = Constant.int(x.type, 1)
    counter = y
    builder.store(counter, pcounter)
    builder.store(result, presult)

    bbcond = cgutils.append_basic_block(builder, ".cond")
    bbbody = cgutils.append_basic_block(builder, ".body")
    bbexit = cgutils.append_basic_block(builder, ".exit")

    del counter
    del result

    builder.branch(bbcond)

    with cgutils.goto_block(builder, bbcond):
        counter = builder.load(pcounter)
        ONE = Constant.int(counter.type, 1)
        ZERO = Constant.null(counter.type)
        builder.store(builder.sub(counter, ONE), pcounter)
        pred = builder.icmp(lc.ICMP_SGT, counter, ZERO)
        builder.cbranch(pred, bbbody, bbexit)

    with cgutils.goto_block(builder, bbbody):
        result = builder.load(presult)
        builder.store(builder.mul(result, x), presult)
        builder.branch(bbcond)

    builder.position_at_end(bbexit)
    return builder.load(presult)


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


def int_abs_impl(context, builder, tys, args):
    [x] = args
    ZERO = Constant.null(x.type)
    ltz = builder.icmp(lc.ICMP_SLT, x, ZERO)
    negated = builder.neg(x)
    return builder.select(ltz, negated, x)


def int_print_impl(context, builder, tys, args):
    [x] = args
    py = context.get_python_api(builder)
    szval = context.cast(builder, x, tys[0], types.intp)
    intobj = py.long_from_ssize_t(szval)
    py.print_object(intobj)
    py.decref(intobj)
    return context.get_dummy_value()


for ty in types.integer_domain:
    builtin(implement('+', ty, ty, ty)(int_add_impl))
    builtin(implement('-', ty, ty, ty)(int_sub_impl))
    builtin(implement('*', ty, ty, ty)(int_mul_impl))
    builtin(implement('==', types.boolean, ty, ty)(int_eq_impl))
    builtin(implement('!=', types.boolean, ty, ty)(int_ne_impl))

    builtin(implement(types.print_type, types.none, ty)(int_print_impl))

for ty in types.unsigned_domain:
    builtin(implement('/?', ty, ty, ty)(int_udiv_impl))
    builtin(implement('%', ty, ty, ty)(int_urem_impl))

for ty in types.signed_domain:
    builtin(implement('/?', ty, ty, ty)(int_sdiv_impl))
    builtin(implement('%', ty, ty, ty)(int_srem_impl))
    builtin(implement('<', types.boolean, ty, ty)(int_slt_impl))
    builtin(implement('<=', types.boolean, ty, ty)(int_sle_impl))
    builtin(implement('>', types.boolean, ty, ty)(int_sgt_impl))
    builtin(implement('>=', types.boolean, ty, ty)(int_sge_impl))
    builtin(implement(types.abs_type, ty, ty)(int_abs_impl))


builtin(implement('**', types.float64, types.float64, types.int32)
        (power_int_impl))


def real_add_impl(context, builder, tys, args):
    return builder.fadd(*args)


def real_sub_impl(context, builder, tys, args):
    return builder.fsub(*args)


def real_mul_impl(context, builder, tys, args):
    return builder.fmul(*args)


def real_div_impl(context, builder, tys, args):
    return builder.fdiv(*args)


def real_divmod(context, builder, x, y):
    assert x.type == y.type
    floatty = x.type

    module = cgutils.get_module(builder)
    fname = ".numba.python.rem.%s" % x.type
    fnty = Type.function(floatty, (floatty, floatty, Type.pointer(floatty)))
    fn = module.get_or_insert_function(fnty, fname)

    if fn.is_declaration:
        fn.linkage = lc.LINKAGE_LINKONCE_ODR
        fnbuilder = lc.Builder.new(fn.append_basic_block('entry'))
        fx, fy, pmod = fn.args
        div, mod = real_divmod_func_body(context, fnbuilder, fx, fy)
        fnbuilder.store(mod, pmod)
        fnbuilder.ret(div)

    pmod = cgutils.alloca_once(builder, floatty)
    quotient = builder.call(fn, (x, y, pmod))
    return quotient, builder.load(pmod)


def real_divmod_func_body(context, builder, vx, wx):
    # Reference Objects/floatobject.c
    #
    # float_divmod(PyObject *v, PyObject *w)
    # {
    #     double vx, wx;
    #     double div, mod, floordiv;
    #     CONVERT_TO_DOUBLE(v, vx);
    #     CONVERT_TO_DOUBLE(w, wx);
    #     mod = fmod(vx, wx);
    #     /* fmod is typically exact, so vx-mod is *mathematically* an
    #        exact multiple of wx.  But this is fp arithmetic, and fp
    #        vx - mod is an approximation; the result is that div may
    #        not be an exact integral value after the division, although
    #        it will always be very close to one.
    #     */
    #     div = (vx - mod) / wx;
    #     if (mod) {
    #         /* ensure the remainder has the same sign as the denominator */
    #         if ((wx < 0) != (mod < 0)) {
    #             mod += wx;
    #             div -= 1.0;
    #         }
    #     }
    #     else {
    #         /* the remainder is zero, and in the presence of signed zeroes
    #            fmod returns different results across platforms; ensure
    #            it has the same sign as the denominator; we'd like to do
    #            "mod = wx * 0.0", but that may get optimized away */
    #         mod *= mod;  /* hide "mod = +0" from optimizer */
    #         if (wx < 0.0)
    #             mod = -mod;
    #     }
    #     /* snap quotient to nearest integral value */
    #     if (div) {
    #         floordiv = floor(div);
    #         if (div - floordiv > 0.5)
    #             floordiv += 1.0;
    #     }
    #     else {
    #         /* div is zero - get the same sign as the true quotient */
    #         div *= div;             /* hide "div = +0" from optimizers */
    #         floordiv = div * vx / wx; /* zero w/ sign of vx/wx */
    #     }
    #     return Py_BuildValue("(dd)", floordiv, mod);
    # }
    pmod = builder.alloca(vx.type)
    pdiv = builder.alloca(vx.type)
    pfloordiv = builder.alloca(vx.type)

    mod = builder.frem(vx, wx)
    div = builder.fdiv(builder.fsub(vx, mod), wx)

    builder.store(mod, pmod)
    builder.store(div, pdiv)

    ZERO = Constant.real(vx.type, 0)
    ONE = Constant.real(vx.type, 1)
    mod_istrue = builder.fcmp(lc.FCMP_ONE, mod, ZERO)
    wx_ltz = builder.fcmp(lc.FCMP_OLT, wx, ZERO)
    mod_ltz = builder.fcmp(lc.FCMP_OLT, mod, ZERO)

    with cgutils.ifthen(builder, mod_istrue):
        wx_ltz_ne_mod_ltz = builder.icmp(lc.ICMP_NE, wx_ltz, mod_ltz)

        with cgutils.ifthen(builder, wx_ltz_ne_mod_ltz):
            mod = builder.fadd(mod, wx)
            div = builder.fsub(div, ONE)
            builder.store(mod, pmod)
            builder.store(div, pdiv)

    del mod
    del div

    with cgutils.ifnot(builder, mod_istrue):
        mod = builder.load(pmod)
        mod = builder.fmul(mod, mod)
        builder.store(mod, pmod)
        del mod

        with cgutils.ifthen(builder, wx_ltz):
            mod = builder.load(pmod)
            mod = builder.fsub(ZERO, mod)
            builder.store(mod, pmod)
            del mod

    div = builder.load(pdiv)
    div_istrue = builder.fcmp(lc.FCMP_ONE, div, ZERO)

    with cgutils.ifthen(builder, div_istrue):
        module = cgutils.get_module(builder)
        floorfn = lc.Function.intrinsic(module, lc.INTR_FLOOR, [wx.type])
        floordiv = builder.call(floorfn, [div])
        floordivdiff = builder.fsub(div, floordiv)
        floordivincr = builder.fadd(floordiv, ONE)
        HALF = Constant.real(wx.type, 0.5)
        pred = builder.fcmp(lc.FCMP_OGT, floordivdiff, HALF)
        floordiv = builder.select(pred, floordivincr, floordiv)
        builder.store(floordiv, pfloordiv)

    with cgutils.ifnot(builder, div_istrue):
        div = builder.fmul(div, div)
        builder.store(div, pdiv)
        floordiv = builder.fdiv(builder.fmul(div, vx), wx)
        builder.store(floordiv, pfloordiv)

    return builder.load(pfloordiv), builder.load(pmod)


def real_mod_impl(context, builder, tys, args):
    x, y = args
    _, rem = real_divmod(context, builder, x, y)
    return rem


def real_power_impl(context, builder, tys, args):
    x, y = args
    module = cgutils.get_module(builder)
    fn = lc.Function.intrinsic(module, lc.INTR_POW, [y.type])
    return builder.call(fn, (x, y))


def real_lt_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OLT, *args)


def real_le_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OLE, *args)


def real_gt_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OGT, *args)


def real_ge_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OGE, *args)


def real_eq_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_OEQ, *args)


def real_ne_impl(context, builder, tys, args):
    return builder.fcmp(lc.FCMP_UNE, *args)


def real_abs_impl(context, builder, tys, args):
    [ty] = tys
    sig = typing.signature(ty, ty)
    impl = context.get_function(math.fabs, sig)
    return impl(context, builder, tys, args)


def real_print_impl(context, builder, tys, args):
    [x] = args
    py = context.get_python_api(builder)
    szval = context.cast(builder, x, tys[0], types.float64)
    intobj = py.float_from_double(szval)
    py.print_object(intobj)
    py.decref(intobj)
    return context.get_dummy_value()


for ty in types.real_domain:
    builtin(implement('+', ty, ty, ty)(real_add_impl))
    builtin(implement('-', ty, ty, ty)(real_sub_impl))
    builtin(implement('*', ty, ty, ty)(real_mul_impl))
    builtin(implement('/?', ty, ty, ty)(real_div_impl))
    builtin(implement('/', ty, ty, ty)(real_div_impl))
    builtin(implement('%', ty, ty, ty)(real_mod_impl))
    builtin(implement('**', ty, ty, ty)(real_power_impl))

    builtin(implement('==', types.boolean, ty, ty)(real_eq_impl))
    builtin(implement('!=', types.boolean, ty, ty)(real_ne_impl))
    builtin(implement('<', types.boolean, ty, ty)(real_lt_impl))
    builtin(implement('<=', types.boolean, ty, ty)(real_le_impl))
    builtin(implement('>', types.boolean, ty, ty)(real_gt_impl))
    builtin(implement('>=', types.boolean, ty, ty)(real_ge_impl))

    builtin(implement(types.abs_type, ty, ty)(real_abs_impl))
    builtin(implement(types.print_type, types.none, ty)(real_print_impl))


class Complex64(cgutils.Structure):
    _fields = [('real', types.float32),
               ('imag', types.float32)]


class Complex128(cgutils.Structure):
    _fields = [('real', types.float64),
               ('imag', types.float64)]


def get_complex_info(ty):
    if ty  == types.complex64:
        cmplxcls = Complex64
        flty = types.float32

    elif ty == types.complex128:
        cmplxcls = Complex128
        flty = types.float64

    return cmplxcls, flty


@builtin_attr
@impl_attribute(types.complex64, "real", types.float32)
def complex64_real_impl(context, builder, typ, value):
    cplx = Complex64(context, builder, value=value)
    return cplx.real


@builtin_attr
@impl_attribute(types.complex128, "real", types.float64)
def complex128_real_impl(context, builder, typ, value):
    cplx = Complex128(context, builder, value=value)
    return cplx.real


@builtin_attr
@impl_attribute(types.complex64, "imag", types.float32)
def complex64_imag_impl(context, builder, typ, value):
    cplx = Complex64(context, builder, value=value)
    return cplx.imag


@builtin_attr
@impl_attribute(types.complex128, "imag", types.float64)
def complex128_imag_impl(context, builder, typ, value):
    cplx = Complex128(context, builder, value=value)
    return cplx.imag


@builtin
@implement("**", types.complex128, types.complex128, types.complex128)
def complex128_power_impl(context, builder, tys, args):
    [ca, cb] = args
    a = Complex128(context, builder, value=ca)
    b = Complex128(context, builder, value=cb)
    c = Complex128(context, builder)
    module = cgutils.get_module(builder)
    pa = a._getvalue()
    pb = b._getvalue()
    pc = c._getvalue()

    # Optimize for square because cpow looses a lot of precsiion
    TWO = context.get_constant(types.float64, 2)
    ZERO = context.get_constant(types.float64, 0)

    b_real_is_two = builder.fcmp(lc.FCMP_OEQ, b.real, TWO)
    b_imag_is_zero = builder.fcmp(lc.FCMP_OEQ, b.imag, ZERO)
    b_is_two = builder.and_(b_real_is_two, b_imag_is_zero)

    with cgutils.ifelse(builder, b_is_two) as (then, otherwise):
        with then:
            # Lower as multiplication
            multimpl = complex_mult_impl(Complex128)
            res = multimpl(context, builder, tys, (ca, ca))
            cres = Complex128(context, builder, value=res)
            c.real = cres.real
            c.imag = cres.imag

        with otherwise:
            # Lower with call to external function
            fnty = Type.function(Type.void(), [pa.type] * 3)
            cpow = module.get_or_insert_function(fnty, name="numba.math.cpow")
            builder.call(cpow, (pa, pb, pc))

    return pc


def complex_add_impl(complexClass):
    def impl(context, builder, tys, args):
        [cx, cy] = args
        x = complexClass(context, builder, value=cx)
        y = complexClass(context, builder, value=cy)
        z = complexClass(context, builder)
        a = x.real
        b = x.imag
        c = y.real
        d = y.imag
        z.real = builder.fadd(a, c)
        z.imag = builder.fadd(b, d)
        return z._getvalue()
    return impl


def complex_sub_impl(complexClass):
    def impl(context, builder, tys, args):
        [cx, cy] = args
        x = complexClass(context, builder, value=cx)
        y = complexClass(context, builder, value=cy)
        z = complexClass(context, builder)
        a = x.real
        b = x.imag
        c = y.real
        d = y.imag
        z.real = builder.fsub(a, c)
        z.imag = builder.fsub(b, d)
        return z._getvalue()
    return impl


def complex_mult_impl(complexClass):
    def impl(context, builder, tys, args):
        """
        (a+bi)(c+di)=(ac-bd)+i(ad+bc)
        """
        [cx, cy] = args
        x = complexClass(context, builder, value=cx)
        y = complexClass(context, builder, value=cy)
        z = complexClass(context, builder)
        a = x.real
        b = x.imag
        c = y.real
        d = y.imag
        ac = builder.fmul(a, c)
        bd = builder.fmul(b, d)
        ad = builder.fmul(a, d)
        bc = builder.fmul(b, c)
        z.real = builder.fsub(ac, bd)
        z.imag = builder.fadd(ad, bc)
        return z._getvalue()
    return impl


for ty, cls in zip([types.complex64, types.complex128],
                   [Complex64, Complex128]):
    builtin(implement("+", ty, ty, ty)(complex_add_impl(cls)))
    builtin(implement("-", ty, ty, ty)(complex_sub_impl(cls)))
    builtin(implement("*", ty, ty, ty)(complex_mult_impl(cls)))


class Slice(cgutils.Structure):
    _fields = [('start', types.intp),
               ('stop', types.intp),
               ('step', types.intp),]


@builtin
@implement(types.slice_type, types.slice3_type, types.intp, types.intp,
           types.intp)
def slice3_impl(context, builder, tys, args):
    start, stop, step = args

    slice3 = Slice(context, builder)
    slice3.start = start
    slice3.stop = stop
    slice3.step = step

    return slice3._getvalue()


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


def make_unituple_iter(tupiter):
    class UniTupleIter(cgutils.Structure):
        _fields = [('index',  types.intp),
                   ('tuple',  tupiter.unituple,)]
    return UniTupleIter


@builtin
@implement(types.range_type, types.range_state32_type, types.int32)
def range1_32_impl(context, builder, tys, args):
    [stop] = args
    state = RangeState32(context, builder)

    state.start = context.get_constant(types.int32, 0)
    state.stop = stop
    state.step = context.get_constant(types.int32, 1)

    return state._getvalue()


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
@implement(types.range_type, types.range_state32_type, types.int32,
           types.int32, types.int32)
def range2_32_impl(context, builder, tys, args):
    [start, stop, step] = args
    state = RangeState32(context, builder)

    state.start = start
    state.stop = stop
    state.step = step

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
@implement(types.range_type, types.range_state64_type, types.int64,
           types.int64)
def range3_64_impl(context, builder, tys, args):
    [start, stop, step] = args
    state = RangeState64(context, builder)

    state.start = start
    state.stop = stop
    state.step = step

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
@implement('getiter', types.Any, types.Kind(types.UniTuple))
def getiter_unituple(context, builder, tys, args):
    [tupty] = tys
    [tup] = args

    tupitercls = context.make_unituple_iter(types.UniTupleIter(tupty))
    iterval = tupitercls(context, builder)
    iterval.index = context.get_constant(types.intp, 0)
    iterval.tuple = tup
    return iterval._getvalue()


@builtin
@implement('iternextsafe', types.Any, types.Kind(types.UniTupleIter))
def iternextsafe_unituple(context, builder, tys, args):
    [tupiterty] = tys
    [tupiter] = args

    tupitercls = context.make_unituple_iter(tupiterty)
    iterval = tupitercls(context, builder, value=tupiter)
    tup = iterval.tuple
    idx = iterval.index

    # TODO lack out-of-bound check
    res = getitem_unituple(context, builder, [tupiterty.unituple, types.intp],
                           [tup, idx])
    iterval.index = builder.add(idx, context.get_constant(types.intp, 1))
    return res


@builtin
@implement('getitem', types.Any, types.Kind(types.UniTuple), types.intp)
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
@implement('getitem', types.Any, types.Kind(types.Array), types.intp)
def getitem_array1d(context, builder, tys, args):
    aryty, _ = tys
    if aryty.ndim != 1:
        # TODO
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)

    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    dataptr = ary.data

    if True or WARPAROUND:  # TODO target flag
        ZERO = context.get_constant(types.intp, 0)
        negative = builder.icmp(lc.ICMP_SLT, idx, ZERO)
        bbnormal = builder.basic_block
        with cgutils.if_unlikely(builder, negative):
            # Index is negative, wraparound
            [nelem] = cgutils.unpack_tuple(builder, ary.shape, 1)
            wrapped = builder.add(nelem, idx)
            bbwrapped = builder.basic_block

        where = builder.phi(idx.type)
        where.add_incoming(idx, bbnormal)
        where.add_incoming(wrapped, bbwrapped)

        ptr = builder.gep(dataptr, [where])
        return builder.load(ptr)
    else:
        # No wraparound
        ptr = builder.gep(dataptr, [idx])
        return builder.load(ptr)


@builtin
@implement('getitem', types.Any, types.Kind(types.Array),
           types.Kind(types.UniTuple))
def getitem_array_unituple(context, builder, tys, args):
    aryty, idxty = tys
    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    # TODO warparound flag
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                   wraparound=True)
    return builder.load(ptr)


@builtin
@implement('setitem', types.none, types.Kind(types.Array), types.intp,
           types.Any)
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
           types.Kind(types.UniTuple), types.Any)
def setitem_array_unituple(context, builder, tys, args):
    aryty, idxty, valty = tys
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other than layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                   wraparound=True)
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
@implement(math.sin, types.float32, types.float32)
def math_sin_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_SIN, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.sin, types.float64, types.float64)
def math_sin_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_SIN, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.cos, types.float32, types.float32)
def math_cos_f32(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float32)
    intr = lc.Function.intrinsic(mod, lc.INTR_COS, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.cos, types.float64, types.float64)
def math_cos_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    lty = context.get_value_type(types.float64)
    intr = lc.Function.intrinsic(mod, lc.INTR_COS, [lty])
    return builder.call(intr, args)


@builtin
@implement(math.tan, types.float32, types.float32)
def math_tan_f32(context, builder, tys, args):
    (val,) = args
    val = context.cast(builder, val, types.float32, types.float64)
    impl = context.get_function(math.tan, types.float64, types.float64)
    res = impl(context, builder, [types.float64], [val])
    return context.cast(builder, res, types.float64, types.float32)


@builtin
@implement(math.tan, types.float64, types.float64)
def math_tan_f64(context, builder, tys, args):
    (val,) = args
    mod = cgutils.get_module(builder)
    fnty = Type.function(Type.double(), [Type.double()])
    fn = mod.get_or_insert_function(fnty, name="tan")
    return builder.call(fn, (val,))


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

#-------------------------------------------------------------------------------


def numpy_unary_ufunc(funckey, asfloat=False):
    def impl(context, builder, tys, args):
        [tyinp, tyout] = tys
        [inp, out] = args
        assert tyinp.dtype == tyout.dtype
        dtype = tyinp.dtype
        ndim = tyinp.ndim

        iary = make_array(tyinp)(context, builder, inp)
        oary = make_array(tyout)(context, builder, out)

        if asfloat:
            sig = typing.signature(types.float64, types.float64)
        else:
            sig = typing.signature(dtype, dtype)

        fnwork = context.get_function(funckey, sig)
        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        shape = cgutils.unpack_tuple(builder, iary.shape, ndim)
        with cgutils.loop_nest(builder, shape, intp=intpty) as indices:
            pi = cgutils.get_item_pointer(builder, tyinp, iary, indices)
            po = cgutils.get_item_pointer(builder, tyout, oary, indices)

            ival = builder.load(pi)
            if asfloat:
                dval = context.cast(builder, ival, dtype, types.float64)
                dres = fnwork(context, builder, [types.float64], [dval])
                res = context.cast(builder, dres, types.float64, dtype)
            else:
                res = fnwork(context, builder, [dtype], [ival])
            builder.store(res, po)

        return out
    return impl


@builtin
@implement(numpy.absolute, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_absolute(context, builder, tys, args):
    imp = numpy_unary_ufunc(types.abs_type)
    return imp(context, builder, tys, args)


@builtin
@implement(numpy.exp, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_exp(context, builder, tys, args):
    imp = numpy_unary_ufunc(math.exp, asfloat=True)
    return imp(context, builder, tys, args)


@builtin
@implement(numpy.sin, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_sin(context, builder, tys, args):
    imp = numpy_unary_ufunc(math.sin, asfloat=True)
    return imp(context, builder, tys, args)


@builtin
@implement(numpy.cos, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_cos(context, builder, tys, args):
    imp = numpy_unary_ufunc(math.cos, asfloat=True)
    return imp(context, builder, tys, args)


@builtin
@implement(numpy.tan, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def numpy_tan(context, builder, tys, args):
    imp = numpy_unary_ufunc(math.tan, asfloat=True)
    return imp(context, builder, tys, args)


def numpy_binary_ufunc(core):
    def impl(context, builder, tys, args):
        [tyvx, tywy, tyout] = tys
        [vx, wy, out] = args
        assert tyvx.dtype == tyout.dtype
        assert tyvx.dtype == tywy.dtype
        dtype = tyvx.dtype
        ndim = tyvx.ndim

        xary = make_array(tyvx)(context, builder, vx)
        yary = make_array(tywy)(context, builder, wy)
        oary = make_array(tyout)(context, builder, out)

        intpty = context.get_value_type(types.intp)

        # TODO handle differing shape by mimicking broadcasting
        loopshape = cgutils.unpack_tuple(builder, xary.shape, ndim)

        xyo_shape = [cgutils.unpack_tuple(builder, ary.shape, ndim)
                     for ary in (xary, yary, oary)]
        xyo_strides = [cgutils.unpack_tuple(builder, ary.strides, ndim)
                       for ary in (xary, yary, oary)]
        xyo_data = [ary.data for ary in (xary, yary, oary)]
        xyo_layout = [ty.layout for ty in (tyvx, tywy, tyout)]

        with cgutils.loop_nest(builder, loopshape, intp=intpty) as indices:
            [px, py, po] = [cgutils.get_item_pointer2(builder,
                                                      data=data, shape=shape,
                                                      strides=strides,
                                                      layout=layout,
                                                      inds=indices)
                            for data, shape, strides, layout
                            in zip(xyo_data, xyo_shape, xyo_strides,
                                   xyo_layout)]

            x = builder.load(px)
            y = builder.load(py)
            res = core(context, builder, (dtype, dtype), (x, y))
            builder.store(res, po)

        return out
    return impl


@builtin
@implement(numpy.add, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_add(context, builder, tys, args):
    def intcore(context, builder, tys, args):
        x, y = args
        return builder.add(x, y)

    def realcore(context, builder, tys, args):
        x, y = args
        return builder.fadd(x, y)

    if tys[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, tys, args)


@builtin
@implement(numpy.subtract, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_sub(context, builder, tys, args):
    def intcore(context, builder, tys, args):
        x, y = args
        return builder.sub(x, y)

    def realcore(context, builder, tys, args):
        x, y = args
        return builder.fsub(x, y)

    if tys[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, tys, args)


@builtin
@implement(numpy.multiply, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_multiply(context, builder, tys, args):
    def intcore(context, builder, tys, args):
        x, y = args
        return builder.mul(x, y)

    def realcore(context, builder, tys, args):
        x, y = args
        return builder.fmul(x, y)

    if tys[0].dtype in types.integer_domain:
        imp = numpy_binary_ufunc(intcore)
    else:
        imp = numpy_binary_ufunc(realcore)

    return imp(context, builder, tys, args)


@builtin
@implement(numpy.divide, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array), types.Kind(types.Array))
def numpy_divide(context, builder, tys, args):
    dtype = tys[0].dtype
    if dtype in types.signed_domain:
        imp = numpy_binary_ufunc(int_sdiv_impl)
    elif dtype in types.unsigned_domain:
        imp = numpy_binary_ufunc(int_udiv_impl)
    else:
        imp = numpy_binary_ufunc(real_div_impl)

    return imp(context, builder, tys, args)
