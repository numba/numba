"""
Calling conventions for Numba-compiled functions.
"""

from collections import namedtuple
from collections.abc import Iterable
import itertools

from llvmlite import ir

from numba.core import types, cgutils
from numba.core.base import PYOBJECT, GENERIC_POINTER


TryStatus = namedtuple('TryStatus', ['in_try', 'excinfo'])


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
        self._return_errcode_raw(builder, RETCODE_EXC, mark_exc=True)

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

    def return_user_exc(self, builder, exc, exc_args=None, loc=None,
                        func_name=None):
        if exc is not None and not issubclass(exc, BaseException):
            raise TypeError("exc should be None or exception class, got %r"
                            % (exc,))
        if exc_args is not None and not isinstance(exc_args, tuple):
            raise TypeError("exc_args should be None or tuple, got %r"
                            % (exc_args,))

        # Build excinfo struct
        if loc is not None:
            fname = loc._raw_function_name()
            if fname is None:
                # could be exec(<string>) or REPL, try func_name
                fname = func_name

            locinfo = (fname, loc.filename, loc.line)
            if None in locinfo:
                locinfo = None
        else:
            locinfo = None

        call_helper = self._get_call_helper(builder)
        exc_id = call_helper._add_exception(exc, exc_args, locinfo)
        self._return_errcode_raw(builder, _const_int(exc_id), mark_exc=True)

    def return_status_propagate(self, builder, status):
        self._return_errcode_raw(builder, status.code)

    def _return_errcode_raw(self, builder, code, mark_exc=False):
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

    def _add_exception(self, exc, exc_args, locinfo):
        """
        Parameters
        ----------
        exc :
            exception type
        exc_args : None or tuple
            exception args
        locinfo : tuple
            location information
        """
        exc_id = len(self.exceptions) + FIRST_USEREXC
        self.exceptions[exc_id] = exc, exc_args, locinfo
        return exc_id

    def get_exception(self, exc_id):
        try:
            return self.exceptions[exc_id]
        except KeyError:
            msg = "unknown error %d in native function" % exc_id
            return SystemError, (msg,)

# The structure type constructed by PythonAPI.serialize_uncached()
# i.e a {i8* pickle_buf, i32 pickle_bufsz, i8* hash_buf}
excinfo_t = ir.LiteralStructType([GENERIC_POINTER, int32_t, GENERIC_POINTER])
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

    def set_static_user_exc(self, builder, exc, exc_args=None, loc=None,
                            func_name=None):
        if exc is not None and not issubclass(exc, BaseException):
            raise TypeError("exc should be None or exception class, got %r"
                            % (exc,))
        if exc_args is not None and not isinstance(exc_args, tuple):
            raise TypeError("exc_args should be None or tuple, got %r"
                            % (exc_args,))
        # None is indicative of no args, set the exc_args to an empty tuple
        # as PyObject_CallObject(exc, exc_args) requires the second argument to
        # be a tuple (or nullptr, but doing this makes it consistent)
        if exc_args is None:
            exc_args = tuple()

        pyapi = self.context.get_python_api(builder)
        # Build excinfo struct
        if loc is not None:
            fname = loc._raw_function_name()
            if fname is None:
                # could be exec(<string>) or REPL, try func_name
                fname = func_name

            locinfo = (fname, loc.filename, loc.line)
            if None in locinfo:
                locinfo = None
        else:
            locinfo = None
        exc = (exc, exc_args, locinfo)
        struct_gv = pyapi.serialize_object(exc)
        excptr = self._get_excinfo_argument(builder.function)
        builder.store(struct_gv, excptr)

    def return_user_exc(self, builder, exc, exc_args=None, loc=None,
                        func_name=None):
        try_info = getattr(builder, '_in_try_block', False)
        self.set_static_user_exc(builder, exc, exc_args=exc_args,
                                   loc=loc, func_name=func_name)
        trystatus = self.check_try_status(builder)
        if try_info:
            # This is a hack for old-style impl.
            # We will branch directly to the exception handler.
            builder.branch(try_info['target'])
        else:
            # Return from the current function
            self._return_errcode_raw(builder, RETCODE_USEREXC, mark_exc=True)

    def _get_try_state(self, builder):
        try:
            return builder.__eh_try_state
        except AttributeError:
            ptr = cgutils.alloca_once(
                builder, cgutils.intp_t, name='try_state', zfill=True,
            )
            builder.__eh_try_state = ptr
            return ptr

    def check_try_status(self, builder):
        try_state_ptr = self._get_try_state(builder)
        try_depth = builder.load(try_state_ptr)
        # try_depth > 0
        in_try = builder.icmp_unsigned('>', try_depth, try_depth.type(0))

        excinfoptr = self._get_excinfo_argument(builder.function)
        excinfo = builder.load(excinfoptr)

        return TryStatus(in_try=in_try, excinfo=excinfo)

    def set_try_status(self, builder):
        try_state_ptr = self._get_try_state(builder)
        # Increment try depth
        old = builder.load(try_state_ptr)
        new = builder.add(old, old.type(1))
        builder.store(new, try_state_ptr)

    def unset_try_status(self, builder):
        try_state_ptr = self._get_try_state(builder)
        # Decrement try depth
        old = builder.load(try_state_ptr)
        new = builder.sub(old, old.type(1))
        builder.store(new, try_state_ptr)

        # Needs to reset the exception state so that the exception handler
        # will run normally.
        excinfoptr = self._get_excinfo_argument(builder.function)
        null = cgutils.get_null_value(excinfoptr.type.pointee)
        builder.store(null, excinfoptr)

    def return_status_propagate(self, builder, status):
        trystatus = self.check_try_status(builder)
        excptr = self._get_excinfo_argument(builder.function)
        builder.store(status.excinfoptr, excptr)
        with builder.if_then(builder.not_(trystatus.in_try)):
            self._return_errcode_raw(builder, status.code, mark_exc=True)

    def _return_errcode_raw(self, builder, code, mark_exc=False):
        ret = builder.ret(code)

        if mark_exc:
            md = builder.module.add_metadata([ir.IntType(1)(1)])
            ret.set_metadata("ret_is_raise", md)

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

        # Add metadata to mark functions that may need NRT
        # thus disabling aggressive refct pruning in removerefctpass.py
        def type_may_always_need_nrt(ty):
            # Returns True if it's a non-Array type that is contains MemInfo
            if not isinstance(ty, types.Array):
                dmm = self.context.data_model_manager
                if dmm[ty].contains_nrt_meminfo():
                    return True
            return False

        args_may_always_need_nrt = any(
            map(type_may_always_need_nrt, fe_argtypes)
        )

        if args_may_always_need_nrt:
            nmd = fn.module.add_named_metadata(
                'numba_args_may_always_need_nrt',
            )
            nmd.add(fn.module.add_metadata([fn]))

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

    def call_function(self, builder, callee, resty, argtys, args,
                      attrs=None):
        """
        Call the Numba-compiled *callee*.
        Parameters:
        -----------
        attrs: LLVM style string or iterable of individual attributes, default
               is None which specifies no attributes. Examples:
               LLVM style string: "noinline fast"
               Equivalent iterable: ("noinline", "fast")
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
        # deal with attrs, it's fine to specify a load in a string like
        # "noinline fast" as per LLVM or equally as an iterable of individual
        # attributes.
        if attrs is None:
            _attrs = ()
        elif isinstance(attrs, Iterable) and not isinstance(attrs, str):
            _attrs = tuple(attrs)
        else:
            raise TypeError("attrs must be an iterable of strings or None")
        code = builder.call(callee, realargs, attrs=_attrs)
        status = self._get_return_status(builder, code,
                                         builder.load(excinfoptr))
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out


class ErrorModel(object):

    def __init__(self, call_conv):
        self.call_conv = call_conv

    def fp_zero_division(self, builder, exc_args=None, loc=None):
        if self.raise_on_fp_zero_division:
            self.call_conv.return_user_exc(builder, ZeroDivisionError, exc_args,
                                           loc)
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
