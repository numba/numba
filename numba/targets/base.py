from __future__ import print_function
from collections import namedtuple, defaultdict
import llvm.core as lc
from llvm.core import Type, Constant
import numpy
from numba import types, utils, cgutils, typing
from numba.pythonapi import PythonAPI
from numba.targets.imputils import (user_function, python_attr_impl, BUILTINS,
                                    BUILTIN_ATTRS)
from numba.targets import builtins


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


Status = namedtuple("Status", ("code", "ok", "err", "exc", "none"))


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
            if (len(ver.signature.args) == len(sig.args) or
                    ver.signature.args[-1] == types.VarArg):
                match = True
                for formal, actual in zip(ver.signature.args, sig.args):
                    if formal == types.VarArg:
                        # vararg argument matches everything
                        break

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
    def __init__(self, typing_context):
        self.typing_context = typing_context

        self.defns = defaultdict(Overloads)
        self.attrs = utils.UniqueDict()
        self.users = utils.UniqueDict()

        self.insert_func_defn(BUILTINS)
        self.insert_attr_defn(BUILTIN_ATTRS)

        # Initialize
        self.init()

    def init(self):
        """
        For subclasses to add initializer
        """
        pass

    def insert_func_defn(self, defns):
        for defn in defns:
            self.defns[defn.key].append(defn)

    def insert_attr_defn(self, defns):
        for attr in defns:
            self.attrs[attr.key] = attr

    def insert_user_function(self, func, fndesc):
        imp = user_function(func, fndesc)
        self.defns[func].append(imp)

        baseclses = (typing.templates.ConcreteTemplate,)
        glbls = dict(key=func, cases=[imp.signature])
        name = "CallTemplate(%s)" % fndesc.mangled_name
        self.users[func] = type(name, baseclses, glbls)

    def insert_class(self, cls, attrs):
        clsty = types.Object(cls)
        for name, vtype in utils.dict_iteritems(attrs):
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
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
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
                isinstance(ty, types.Dispatcher) or
                isinstance(ty, types.Object)):
            return self.get_dummy_type()
        elif isinstance(ty, types.Optional):
            return self.get_value_type(ty.type)
        elif ty == types.complex64:
            stty = self.get_struct_type(builtins.Complex64)
            return Type.pointer(stty)
        elif ty == types.complex128:
            stty = self.get_struct_type(builtins.Complex128)
            return Type.pointer(stty)
        elif ty == types.range_state32_type:
            stty = self.get_struct_type(builtins.RangeState32)
            return Type.pointer(stty)
        elif ty == types.range_iter32_type:
            stty = self.get_struct_type(builtins.RangeIter32)
            return Type.pointer(stty)
        elif ty == types.range_state64_type:
            stty = self.get_struct_type(builtins.RangeState64)
            return Type.pointer(stty)
        elif ty == types.range_iter64_type:
            stty = self.get_struct_type(builtins.RangeIter64)
            return Type.pointer(stty)
        elif ty == types.slice3_type:
            stty = self.get_struct_type(builtins.Slice)
            return Type.pointer(stty)
        elif isinstance(ty, types.Array):
            stty = self.get_struct_type(self.make_array(ty))
            return Type.pointer(stty)
        elif isinstance(ty, types.CPointer):
            dty = self.get_value_type(ty.dtype)
            if self.is_struct_type(ty.dtype):
                return dty
            else:
                return Type.pointer(dty)
        elif isinstance(ty, types.UniTuple):
            dty = self.get_value_type(ty.dtype)
            return Type.array(dty, ty.count)
        elif isinstance(ty, types.Tuple):
            dtys = [self.get_value_type(t) for t in ty]
            return Type.struct(dtys)
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

        elif isinstance(ty, types.UniTuple):
            consts = [self.get_constant(ty.dtype, v) for v in val]
            return Constant.array(consts[0].type, consts)

        raise NotImplementedError(ty)

    def get_constant_undef(self, ty):
        lty = self.get_value_type(ty)
        return Constant.undef(lty)

    def get_constant_null(self, ty):
        lty = self.get_value_type(ty)
        return Constant.null(lty)

    def get_function(self, fn, sig):
        if isinstance(fn, types.Function):
            key = fn.template.key
            overloads = self.defns[key]
        elif isinstance(fn, types.Dispatcher):
            key = fn.overloaded.get_overload(sig.args)
            overloads = self.defns[key]
        else:
            key = fn
            overloads = self.defns[key]
        try:
            return _wrap_impl(overloads.find(sig), self, sig)
        except NotImplementedError:
            raise Exception("No definition for lowering %s%s" % (key, sig))

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

    def get_argument_value(self, builder, ty, val):
        if ty == types.boolean:
            return builder.trunc(val, self.get_value_type(ty))
        return val

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

    def return_errcode_propagate(self, builder, code):
        builder.ret(code)

    def return_exc(self, builder):
        builder.ret(RETCODE_EXC)

    def cast(self, builder, val, fromty, toty):
        if fromty == toty or toty == types.Any or isinstance(toty, types.Kind):
            return val

        elif ((fromty in types.unsigned_domain and
               toty in types.signed_domain) or
              (fromty in types.integer_domain and
               toty in types.unsigned_domain)):
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

        elif fromty in types.real_domain and toty in types.complex_domain:
            if fromty == types.float32:
                if toty == types.complex128:
                    real = self.cast(builder, val, fromty, types.float64)
                else:
                    real = val

            elif fromty == types.float64:
                if toty == types.complex64:
                    real = self.cast(builder, val, fromty, types.float32)
                else:
                    real = val

            if toty == types.complex128:
                imag = self.get_constant(types.float64, 0)
            elif toty == types.complex64:
                imag = self.get_constant(types.float32, 0)
            else:
                raise Exception("unreachable")

            cmplx = self.make_complex(toty)(self, builder)
            cmplx.real = real
            cmplx.imag = imag
            return cmplx._getvalue()

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
            cmplxcls, flty = builtins.get_complex_info(toty)
            cmpl = cmplxcls(self, builder)
            cmpl.real = self.cast(builder, val, fromty, flty)
            cmpl.imag = self.get_constant(flty, 0)
            return cmpl._getvalue()

        elif fromty in types.complex_domain and toty in types.complex_domain:
            srccls, srcty = builtins.get_complex_info(fromty)
            dstcls, dstty = builtins.get_complex_info(toty)

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

        elif toty == types.boolean:
            return self.is_true(builder, fromty, val)

        raise NotImplementedError("cast", val, fromty, toty)

    def is_true(self, builder, typ, val):
        if typ in types.integer_domain:
            return builder.icmp(lc.ICMP_NE, val, Constant.null(val.type))
        elif typ in types.real_domain:
            return builder.fcmp(lc.FCMP_ONE, val, Constant.real(val.type, 0))
        elif typ in types.complex_domain:
            cmplx = self.make_complex(typ)(self, builder, val)
            fty = types.float32 if typ == types.complex64 else types.float64
            real_istrue = self.is_true(builder, fty, cmplx.real)
            imag_istrue = self.is_true(builder, fty, cmplx.imag)
            return builder.or_(real_istrue, imag_istrue)
        raise NotImplementedError("is_true", val, typ)

    def call_function(self, builder, callee, argtys, args):
        retty = callee.args[0].type.pointee
        retval = cgutils.alloca_once(builder, retty)
        realargs = [retval] + [self.get_argument_value(builder, t, a)
                               for t, a in zip(argtys, args)]
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
        exc = builder.icmp(lc.ICMP_EQ, code, RETCODE_EXC)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)

        status = Status(code=code, ok=ok, err=err, exc=exc, none=none)
        return status

    def call_function_pointer(self, builder, funcptr, signature, args):
        retty = self.get_value_type(signature.return_type)
        fnty = Type.function(retty, [a.type for a in args])
        fnptrty = Type.pointer(fnty)
        addr = self.get_constant(types.intp, funcptr)
        ptr = builder.inttoptr(addr, fnptrty)
        return builder.call(ptr, args)

    def call_class_method(self, builder, func, signature, args):
        api = self.get_python_api(builder)
        tys = signature.args
        retty = signature.return_type
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

    def debug_print(self, builder, text):
        mod = cgutils.get_module(builder)
        cstr = self.insert_const_string(mod, str(text))
        self.print_string(builder, cstr)

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
        return builtins.make_array(typ)

    def make_complex(self, typ):
        cls, _ = builtins.get_complex_info(typ)
        return cls

    def make_unituple_iter(self, typ):
        return builtins.make_unituple_iter(typ)

    def make_constant_array(self, builder, typ, ary):
        assert typ.layout == 'C'                # assumed in typeinfer.py
        ary = numpy.ascontiguousarray(ary)
        flat = ary.flatten()

        # Handle data
        if self.is_struct_type(typ.dtype):
            # FIXME
            raise TypeError("Do not support structure dtype as constant "
                            "array, yet.")

        values = [self.get_constant(typ.dtype, flat[i])
                  for i in range(flat.size)]

        lldtype = values[0].type
        consts = Constant.array(lldtype, values)

        module = cgutils.get_module(builder)

        data = module.add_global_variable(consts.type, name=".const.array"
                                                            ".data")
        data.linkage = lc.LINKAGE_INTERNAL
        data.global_constant = True
        data.initializer = consts

        # Handle shape
        llintp = self.get_value_type(types.intp)
        shapevals = [self.get_constant(types.intp, s) for s in ary.shape]
        cshape = Constant.array(llintp, shapevals)


        # Handle strides
        stridevals = [self.get_constant(types.intp, s) for s in ary.strides]
        cstrides = Constant.array(llintp, stridevals)

        # Create array structure
        cary = self.make_array(typ)(self, builder)
        cary.data = builder.bitcast(data, cary.data.type)
        cary.shape = cshape
        cary.strides = cstrides
        return cary._getvalue()


def _wrap_impl(imp, context, sig):
    def wrapped(builder, args):
        return imp(context, builder, sig, args)
    return wrapped
