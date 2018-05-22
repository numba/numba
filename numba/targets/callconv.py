"""
Calling conventions for Numba-compiled functions.
"""

from collections import namedtuple
import itertools

from llvmlite import ir as ir

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
                     # If the generator exited with StopIteration
                     "is_stop_iteration",
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
    return ir.Constant(errcode_t, code)

RETCODE_OK = _const_int(0)
RETCODE_EXC = _const_int(-1)
RETCODE_NONE = _const_int(-2)
# StopIteration
RETCODE_STOPIT = _const_int(-3)

FIRST_USEREXC = 1

RETCODE_USEREXC = _const_int(FIRST_USEREXC)




class BaseCallConv(object):

    def __init__(self, context):
        self.context = context

    def return_optional_value(self, builder, retty, valty, value):
        if valty == types.none:
            # Value is none
            self.return_native_none(builder)

        elif retty == valty:
            # Value is an optional, need a runtime switch
            optval = self.context.make_helper(builder, retty, value=value)

            validbit = cgutils.as_bool_bit(builder, optval.valid)
            with builder.if_then(validbit):
                retval = self.context.get_return_value(builder, retty.type,
                                                       optval.data)
                self.return_value(builder, retval)

            self.return_native_none(builder)

        elif not isinstance(valty, types.Optional):
            # Value is not an optional, need a cast
            if valty != retty.type:
                value = self.context.cast(builder, value, fromty=valty,
                                          toty=retty.type)
            retval = self.context.get_return_value(builder, retty.type, value)
            self.return_value(builder, retval)

        else:
            raise NotImplementedError("returning {0} for {1}".format(valty,
                                                                     retty))

    def return_native_none(self, builder):
        self._return_errcode_raw(builder, RETCODE_NONE)

    def return_exc(self, builder):
        self._return_errcode_raw(builder, RETCODE_EXC)

    def return_stop_iteration(self, builder):
        self._return_errcode_raw(builder, RETCODE_STOPIT)

    def get_return_type(self, ty):
        """
        Get the actual type of the return argument for Numba type *ty*.
        """
        restype = self.context.data_model_manager[ty].get_return_type()
        return restype.as_pointer()

    def init_call_helper(self, builder):
        """
        Initialize and return a call helper object for the given builder.
        """
        ch = self._make_call_helper(builder)
        builder.__call_helper = ch
        return ch

    def _get_call_helper(self, builder):
        return builder.__call_helper

    def raise_error(self, builder, api, status):
        """
        Given a non-ok *status*, raise the corresponding Python exception.
        """
        bbend = builder.function.append_basic_block()

        with builder.if_then(status.is_user_exc):
            # Unserialize user exception.
            # Make sure another error may not interfere.
            api.err_clear()
            exc = api.unserialize(status.excinfoptr)
            with cgutils.if_likely(builder,
                                   cgutils.is_not_null(builder, exc)):
                api.raise_object(exc)  # steals ref
            builder.branch(bbend)

        with builder.if_then(status.is_stop_iteration):
            api.err_set_none("PyExc_StopIteration")
            builder.branch(bbend)

        with builder.if_then(status.is_python_exc):
            # Error already raised => nothing to do
            builder.branch(bbend)

        api.err_set_string("PyExc_SystemError",
                           "unknown error when calling native function")
        builder.branch(bbend)

        builder.position_at_end(bbend)

    def decode_arguments(self, builder, argtypes, func):
        """
        Get the decoded (unpacked) Python arguments with *argtypes*
        from LLVM function *func*.  A tuple of LLVM values is returned.
        """
        raw_args = self.get_arguments(func)
        arginfo = self._get_arg_packer(argtypes)
        return arginfo.from_arguments(builder, raw_args)

    def _fix_argtypes(self, argtypes):
        """
        Fix argument types, removing any omitted arguments.
        """
        return tuple(ty for ty in argtypes
                     if not isinstance(ty, types.Omitted))

    def _get_arg_packer(self, argtypes):
        """
        Get an argument packer for the given argument types.
        """
        return self.context.get_arg_packer(argtypes)


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
        retptr = builder.function.args[0]
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(self, builder, exc, exc_args=None):
        if exc is not None and not issubclass(exc, BaseException):
            raise TypeError("exc should be None or exception class, got %r"
                            % (exc,))
        if exc_args is not None and not isinstance(exc_args, tuple):
            raise TypeError("exc_args should be None or tuple, got %r"
                            % (exc_args,))
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
        norm = builder.icmp_signed('==', code, RETCODE_OK)
        none = builder.icmp_signed('==', code, RETCODE_NONE)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)
        exc = builder.icmp_signed('==', code, RETCODE_EXC)
        is_stop_iteration = builder.icmp_signed('==', code, RETCODE_STOPIT)
        is_user_exc = builder.icmp_signed('>=', code, RETCODE_USEREXC)

        status = Status(code=code,
                        is_ok=ok,
                        is_error=err,
                        is_python_exc=exc,
                        is_none=none,
                        is_user_exc=is_user_exc,
                        is_stop_iteration=is_stop_iteration,
                        excinfoptr=None)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        resptr = self.get_return_type(restype)
        fnty = ir.FunctionType(errcode_t, [resptr] + argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn),
                             ['arg.' + a for a in args])
        fn.args[0].name = ".ret"
        return fn

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args[1:]

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(cgutils.get_null_value(retty), retvaltmp)

        arginfo = self._get_arg_packer(argtys)
        args = arginfo.as_arguments(builder, args)
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

        retcode_t (<Python return type>*, excinfo **, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants.
    If RETCODE_USEREXC, the exception info pointer will be filled with
    a pointer to a constant struct describing the raised exception.

    Caller is responsible for allocating slots for the return value
    and the exception info pointer (passed as first and second arguments,
    respectively).
    """
    _status_ids = itertools.count(1)

    def _make_call_helper(self, builder):
        return None

    def return_value(self, builder, retval):
        retptr = self._get_return_argument(builder.function)
        assert retval.type == retptr.type.pointee, \
            (str(retval.type), str(retptr.type.pointee))
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(self, builder, exc, exc_args=None):
        if exc is not None and not issubclass(exc, BaseException):
            raise TypeError("exc should be None or exception class, got %r"
                            % (exc,))
        if exc_args is not None and not isinstance(exc_args, tuple):
            raise TypeError("exc_args should be None or tuple, got %r"
                            % (exc_args,))
        pyapi = self.context.get_python_api(builder)
        # Build excinfo struct
        if exc_args is not None:
            exc = (exc, exc_args)
        struct_gv = pyapi.serialize_object(exc)
        excptr = self._get_excinfo_argument(builder.function)
        builder.store(struct_gv, excptr)
        self._return_errcode_raw(builder, RETCODE_USEREXC)

    def return_status_propagate(self, builder, status):
        excptr = self._get_excinfo_argument(builder.function)
        builder.store(status.excinfoptr, excptr)
        self._return_errcode_raw(builder, status.code)

    def _return_errcode_raw(self, builder, code):
        builder.ret(code)

    def _get_return_status(self, builder, code, excinfoptr):
        """
        Given a return *code* and *excinfoptr*, get a Status instance.
        """
        norm = builder.icmp_signed('==', code, RETCODE_OK)
        none = builder.icmp_signed('==', code, RETCODE_NONE)
        exc = builder.icmp_signed('==', code, RETCODE_EXC)
        is_stop_iteration = builder.icmp_signed('==', code, RETCODE_STOPIT)
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
                        is_stop_iteration=is_stop_iteration,
                        excinfoptr=excinfoptr)
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        resptr = self.get_return_type(restype)
        fnty = ir.FunctionType(errcode_t,
                               [resptr, ir.PointerType(excinfo_ptr_t)]
                               + argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names of function arguments, and add useful attributes to them.
        """
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn),
                             ['arg.' + a for a in args])
        retarg = self._get_return_argument(fn)
        retarg.name = "retptr"
        retarg.add_attribute("nocapture")
        retarg.add_attribute("noalias")
        excarg = self._get_excinfo_argument(fn)
        excarg.name = "excinfo"
        excarg.add_attribute("nocapture")
        excarg.add_attribute("noalias")

        if noalias:
            args = self.get_arguments(fn)
            for a in args:
                if isinstance(a.type, ir.PointerType):
                    a.add_attribute("nocapture")
                    a.add_attribute("noalias")
        return fn

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args[2:]

    def _get_return_argument(self, func):
        return func.args[0]

    def _get_excinfo_argument(self, func):
        return func.args[1]

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        # XXX better fix for callees that are not function values
        #     (pointers to function; thus have no `.args` attribute)
        retty = self._get_return_argument(callee.function_type).pointee

        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value to zeros
        builder.store(cgutils.get_null_value(retty), retvaltmp)

        excinfoptr = cgutils.alloca_once(builder, ir.PointerType(excinfo_t),
                                         name="excinfo")

        arginfo = self._get_arg_packer(argtys)
        args = list(arginfo.as_arguments(builder, args))
        realargs = [retvaltmp, excinfoptr] + args
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code,
                                         builder.load(excinfoptr))
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out


class ErrorModel(object):

    def __init__(self, call_conv):
        self.call_conv = call_conv

    def fp_zero_division(self, builder, exc_args=None):
        if self.raise_on_fp_zero_division:
            self.call_conv.return_user_exc(builder, ZeroDivisionError, exc_args)
            return True
        else:
            return False


class PythonErrorModel(ErrorModel):
    """
    The Python error model.  Any invalid FP input raises an exception.
    """
    raise_on_fp_zero_division = True


class NumpyErrorModel(ErrorModel):
    """
    In the Numpy error model, floating-point errors don't raise an
    exception.  The FPU exception state is inspected by Numpy at the
    end of a ufunc's execution and a warning is raised if appropriate.

    Note there's no easy way to set the FPU exception state from LLVM.
    Instructions known to set an FP exception can be optimized away:
        https://llvm.org/bugs/show_bug.cgi?id=6050
        http://lists.llvm.org/pipermail/llvm-dev/2014-September/076918.html
        http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140929/237997.html
    """
    raise_on_fp_zero_division = False


error_models = {
    'python': PythonErrorModel,
    'numpy': NumpyErrorModel,
    }


def create_error_model(model_name, context):
    """
    Create an error model instance for the given target context.
    """
    return error_models[model_name](context.call_conv)
