"""
Calling conventions for Numba-compiled functions.
"""

from collections import namedtuple
import itertools

from llvmlite import ir as ir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant

from numba import cgutils, errcode, types
from .base import PYOBJECT, GENERIC_POINTER


Status = namedtuple("Status",
                    ("code", "ok", "err", "exc", "none", "excinfoptr"))

int32_t = ir.IntType(32)
errcode_t = int32_t

def _const_int(code):
    return Constant.int_signextend(errcode_t, code)

RETCODE_OK = _const_int(0)
RETCODE_EXC = _const_int(-1)
RETCODE_NONE = _const_int(-2)
RETCODE_USEREXC = _const_int(errcode.ERROR_COUNT)


class BaseCallConv(object):

    def __init__(self, context):
        self.context = context

    def return_optional_value(self, builder, retty, valty, value):
        if valty == types.none:
            self.return_native_none(builder)

        elif retty == valty:
            optcls = self.context.make_optional(retty)
            optval = optcls(self.context, builder, value=value)

            validbit = builder.trunc(optval.valid, lc.Type.int(1))
            with cgutils.ifthen(builder, validbit):
                self.return_value(builder, optval.data)

            self.return_native_none(builder)

        elif not isinstance(valty, types.Optional):
            if valty != retty.type:
                value = self.context.cast(builder, value, fromty=valty,
                                          toty=retty.type)
            self.return_value(builder, value)

        else:
            raise NotImplementedError("returning {0} for {1}".format(valty,
                                                                     retty))

    def return_native_none(self, builder):
        self._return_errcode_raw(builder, RETCODE_NONE)

    def return_errcode(self, builder, code):
        assert code > 0 and code < errcode.ERROR_COUNT
        self._return_errcode_raw(builder, _const_int(code))

    def return_errcode_propagate(self, builder, code):
        self._return_errcode_raw(builder, code)

    def return_exc(self, builder):
        self._return_errcode_raw(builder, RETCODE_EXC)

    def get_return_type(self, ty):
        """
        Get the actual type of the return argument for Numba type *ty*.
        """
        if isinstance(ty, types.Optional):
            return self.get_return_type(ty.type)
        elif self.context.is_struct_type(ty):
            # Argument type is already a pointer
            return self.context.get_argument_type(ty)
        else:
            argty = self.context.get_argument_type(ty)
            return Type.pointer(argty)


class MinimalCallConv(BaseCallConv):
    """
    A minimal calling convention, suitable for e.g. GPU targets.
    """

    def return_value(self, builder, retval):
        fn = cgutils.get_function(builder)
        retptr = fn.args[0]
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        builder.ret(RETCODE_OK)

    def return_user_exc(self, builder, exceptions, exc, exc_args=None):
        assert (exc is None or issubclass(exc, BaseException)), exc
        exc_id = len(exceptions) + errcode.ERROR_COUNT
        exceptions[exc_id] = exc, exc_args
        self._return_errcode_raw(builder, _const_int(exc_id))

    def _return_errcode_raw(self, builder, code):
        builder.ret(code)

    def _get_return_status(self, builder, code):
        """
        Given a return *code*, get a Status instance.
        """
        norm = builder.icmp(lc.ICMP_EQ, code, RETCODE_OK)
        none = builder.icmp(lc.ICMP_EQ, code, RETCODE_NONE)
        exc = builder.icmp(lc.ICMP_EQ, code, RETCODE_EXC)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)

        status = Status(code=code, ok=ok, err=err, exc=exc, none=none,
                        excinfoptr=None)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        Some parameters can be added or shuffled around.
        This is kept in sync with call_function() and get_arguments().

        Calling Convention
        ------------------
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.

        Actual arguments starts at the 2nd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.context.get_argument_type(aty) for aty in argtypes]
        resptr = self.get_return_type(restype)
        fnty = Type.function(Type.int(), [resptr] + argtypes)
        return fnty

    def decorate_function(self, fn, args):
        """
        Set names of function arguments.
        """
        for ak, av in zip(args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        fn.args[0].name = ".ret"
        return fn

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        See get_function_type() for the calling convention.
        """
        return func.args[1:]

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*, using the same calling
        convention as in get_function_type().
        """
        assert env is None
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(lc.Constant.null(retty), retvaltmp)
        args = [self.context.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out


excinfo_t = ir.LiteralStructType([GENERIC_POINTER, int32_t])
excinfo_ptr_t = ir.PointerType(excinfo_t)


class CPUCallConv(BaseCallConv):
    """
    The calling convention for CPU targets, adding an environment argument.
    """
    _status_ids = itertools.count(1)

    def return_value(self, builder, retval):
        fn = cgutils.get_function(builder)
        retptr = self._get_return_argument(fn)
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(self, builder, exceptions, exc, exc_args=None):
        fn = cgutils.get_function(builder)
        pyapi = self.context.get_python_api(builder)
        assert (exc is None or issubclass(exc, BaseException)), exc
        exc_id = len(exceptions) + errcode.ERROR_COUNT
        exceptions[exc_id] = exc, exc_args
        # Build excinfo struct
        if exc_args is not None:
            exc = (exc, exc_args)
        struct_gv = pyapi.serialize_object(exc)
        struct_gv.linkage = 'private'
        builder.store(struct_gv, self._get_excinfo_argument(fn))
        self._return_errcode_raw(builder, _const_int(exc_id))

    def _return_errcode_raw(self, builder, code):
        builder.ret(code)

    def _get_return_status(self, builder, code, excinfoptr):
        """
        Given a return *code* and *excinfoptr*, get a Status instance.
        """
        norm = builder.icmp(lc.ICMP_EQ, code, RETCODE_OK)
        none = builder.icmp(lc.ICMP_EQ, code, RETCODE_NONE)
        exc = builder.icmp(lc.ICMP_EQ, code, RETCODE_EXC)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)
        is_userexc = builder.icmp_signed('>=', code, RETCODE_USEREXC)
        excinfoptr = builder.select(is_userexc, excinfoptr,
                                    ir.Constant(excinfo_ptr_t, ir.Undefined))

        status = Status(code=code, ok=ok, err=err, exc=exc, none=none,
                        excinfoptr=excinfoptr)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        Some parameters can be added or shuffled around.
        This is kept in sync with call_function() and get_arguments().

        Calling Convention
        ------------------
        (Same return value convention as BaseContext target.)
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.

        The 2nd argument is a _dynfunc.Environment object.
        It MUST NOT be used if the function is in nopython mode.

        Actual arguments starts at the 3rd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.context.get_argument_type(aty)
                    for aty in argtypes]
        resptr = self.get_return_type(restype)
        fnty = lc.Type.function(errcode_t,
                                [resptr, ir.PointerType(excinfo_ptr_t), PYOBJECT]
                                + argtypes)
        return fnty

    def decorate_function(self, fn, args):
        """
        Set names of function arguments.
        """
        for ak, av in zip(args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        self._get_return_argument(fn).name = "retptr"
        self._get_excinfo_argument(fn).name = "excinfo"
        self.get_env_argument(fn).name = "env"
        return fn

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        See get_function_type() for the calling convention.
        """
        return func.args[3:]

    def get_env_argument(self, func):
        """
        Get the environment argument of LLVM *func*.
        """
        return func.args[2]

    def _get_return_argument(self, func):
        return func.args[0]

    def _get_excinfo_argument(self, func):
        return func.args[1]

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*, using the same calling
        convention as in get_function_type().
        """
        if env is None:
            # This only works with functions that don't use the environment
            # (nopython functions).
            env = lc.Constant.null(PYOBJECT)
        retty = self._get_return_argument(callee).type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value to zeros
        builder.store(lc.Constant.null(retty), retvaltmp)

        excinfoptr = cgutils.alloca_once(builder, ir.PointerType(excinfo_t),
                                         name="excinfo")

        args = [self.context.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        realargs = [retvaltmp, excinfoptr, env] + args
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code,
                                         builder.load(excinfoptr))
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out

