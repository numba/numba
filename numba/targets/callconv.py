"""
Calling conventions for Numba-compiled functions.
"""

from collections import namedtuple
import itertools

from llvmlite import ir as ir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type, Constant

from numba import cgutils, types
from .base import PYOBJECT, GENERIC_POINTER


Status = namedtuple("Status",
                    ("code",
                     # If the function returned ok (a value or None)
                     "is_ok",
                     # If the function returned None
                     "is_none",
                     # If the function errored out (== not is_ok)
                     "is_error",
                     # If the function errored with an already set exception
                     "is_python_exc",
                     # If the function errored with a user exception
                     "is_user_exc",
                     # The pointer to the exception info structure (for user exceptions)
                     "excinfoptr",
                     ))

int32_t = ir.IntType(32)
errcode_t = int32_t

def _const_int(code):
    return Constant.int_signextend(errcode_t, code)

RETCODE_OK = _const_int(0)
RETCODE_EXC = _const_int(-1)
RETCODE_NONE = _const_int(-2)

FIRST_USEREXC = 1

RETCODE_USEREXC = _const_int(FIRST_USEREXC)




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

    def init_call_helper(self, builder):
        """
        Initialize and return a call helper object for the given builder.
        """
        ch = self._make_call_helper(builder)
        builder.__call_helper = ch
        return ch

    def _get_call_helper(self, builder):
        return builder.__call_helper


class MinimalCallConv(BaseCallConv):
    """
    A minimal calling convention, suitable for e.g. GPU targets.
    The implemented function signature is:

        retcode_t (<Python return type>*, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants or a
    function-specific user exception id (>= RETCODE_USEREXC).

    Caller is responsible for allocating a slot for the return value
    (passed as a pointer in the first argument).
    """

    def _make_call_helper(self, builder):
        return _MinimalCallHelper()

    def return_value(self, builder, retval):
        fn = cgutils.get_function(builder)
        retptr = fn.args[0]
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(self, builder, exc, exc_args=None):
        assert (exc is None or issubclass(exc, BaseException)), exc
        assert (exc_args is None or isinstance(exc_args, tuple)), exc_args
        call_helper = self._get_call_helper(builder)
        exc_id = call_helper._add_exception(exc, exc_args)
        self._return_errcode_raw(builder, _const_int(exc_id))

    def return_status_propagate(self, builder, status):
        self._return_errcode_raw(builder, status.code)

    def _return_errcode_raw(self, builder, code):
        if isinstance(code, int):
            code = _const_int(code)
        builder.ret(code)

    def _get_return_status(self, builder, code):
        """
        Given a return *code*, get a Status instance.
        """
        norm = builder.icmp(lc.ICMP_EQ, code, RETCODE_OK)
        none = builder.icmp(lc.ICMP_EQ, code, RETCODE_NONE)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)
        exc = builder.icmp(lc.ICMP_EQ, code, RETCODE_EXC)
        is_user_exc = builder.icmp_signed('>=', code, RETCODE_USEREXC)

        status = Status(code=code,
                        is_ok=ok,
                        is_error=err,
                        is_python_exc=exc,
                        is_none=none,
                        is_user_exc=is_user_exc,
                        excinfoptr=None)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
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
        Call the Numba-compiled *callee*.
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


class _MinimalCallHelper(object):
    """
    A call helper object for the "minimal" calling convention.
    User exceptions are represented as integer codes and stored in
    a mapping for retrieval from the caller.
    """

    def __init__(self):
        self.exceptions = {}

    def _add_exception(self, exc, exc_args):
        exc_id = len(self.exceptions) + FIRST_USEREXC
        self.exceptions[exc_id] = exc, exc_args
        return exc_id

    def get_exception(self, exc_id):
        try:
            return self.exceptions[exc_id]
        except KeyError:
            msg = "unknown error %d in native function" % exc_id
            return SystemError, (msg,)


excinfo_t = ir.LiteralStructType([GENERIC_POINTER, int32_t])
excinfo_ptr_t = ir.PointerType(excinfo_t)


class CPUCallConv(BaseCallConv):
    """
    The calling convention for CPU targets.
    The implemented function signature is:

        retcode_t (<Python return type>*, excinfo **, env *, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants.
    If RETCODE_USEREXC, the exception info pointer will be filled with
    a pointer to a constant struct describing the raised exception.

    Caller is responsible for allocating slots for the return value
    and the exception info pointer (passed as first and second arguments,
    respectively).

    The third argument (env *) is a _dynfunc.Environment object, used
    only for object mode functions.
    """
    _status_ids = itertools.count(1)

    def _make_call_helper(self, builder):
        return None

    def return_value(self, builder, retval):
        fn = cgutils.get_function(builder)
        retptr = self._get_return_argument(fn)
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(self, builder, exc, exc_args=None):
        assert (exc is None or issubclass(exc, BaseException)), exc
        assert (exc_args is None or isinstance(exc_args, tuple)), exc_args
        fn = cgutils.get_function(builder)
        pyapi = self.context.get_python_api(builder)
        # Build excinfo struct
        if exc_args is not None:
            exc = (exc, exc_args)
        struct_gv = pyapi.serialize_object(exc)
        builder.store(struct_gv, self._get_excinfo_argument(fn))
        self._return_errcode_raw(builder, RETCODE_USEREXC)

    def return_status_propagate(self, builder, status):
        fn = cgutils.get_function(builder)
        builder.store(status.excinfoptr, self._get_excinfo_argument(fn))
        self._return_errcode_raw(builder, status.code)

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
        is_user_exc = builder.icmp_signed('>=', code, RETCODE_USEREXC)
        excinfoptr = builder.select(is_user_exc, excinfoptr,
                                    ir.Constant(excinfo_ptr_t, ir.Undefined))

        status = Status(code=code,
                        is_ok=ok,
                        is_error=err,
                        is_python_exc=exc,
                        is_none=none,
                        is_user_exc=is_user_exc,
                        excinfoptr=excinfoptr)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
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
        Call the Numba-compiled *callee*.
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
