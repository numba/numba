from __future__ import print_function, division, absolute_import

from llvmlite.llvmpy.core import Type, Builder, Constant
import llvmlite.llvmpy.core as lc

from numba import types, cgutils


class _ArgManager(object):
    """
    A utility class to handle argument unboxing and cleanup
    """
    def __init__(self, context, builder, api, endblk, nargs):
        self.context = context
        self.builder = builder
        self.api = api
        self.arg_count = 0  # how many function arguments have been processed
        self.cleanups = []
        self.nextblk = endblk

    def add_arg(self, obj, ty):
        """
        Unbox argument and emit code that handles any error during unboxing.
        Args are cleaned up in reverse order of the parameter list, and
        cleanup begins as soon as unboxing of any argument fails. E.g. failure
        on arg2 will result in control flow going through:

            arg2.err -> arg1.err -> arg0.err -> arg.end (returns)
        """
        # Unbox argument
        native = self.api.to_native_value(self.builder.load(obj), ty)

        # If an error occurred, go to the cleanup block for the previous argument.
        with cgutils.if_unlikely(self.builder, native.is_error):
            self.builder.branch(self.nextblk)

        # Write the cleanup block for this argument
        cleanupblk = cgutils.append_basic_block(self.builder,
                                                "arg%d.err" % self.arg_count)
        with cgutils.goto_block(self.builder, cleanupblk):
            # NRT cleanup

            if self.context.enable_nrt:
                def nrt_cleanup():
                    self.context.nrt_decref(self.builder, ty, native.value)
                nrt_cleanup()
                self.cleanups.append(nrt_cleanup)

            if native.cleanup is not None:
                native.cleanup()
                self.cleanups.append(native.cleanup)
            # Go to next cleanup block
            self.builder.branch(self.nextblk)

        self.nextblk = cleanupblk
        self.arg_count += 1
        return native.value

    def emit_cleanup(self):
        """Emit the cleanup code after we are done with the arguments
        """
        for dtor in self.cleanups:
            dtor()


class _GilManager(object):
    """
    A utility class to handle releasing the GIL and then re-acquiring it
    again.
    """

    def __init__(self, builder, api, argman):
        self.builder = builder
        self.api = api
        self.argman = argman
        self.thread_state = api.save_thread()

    def emit_cleanup(self):
        self.api.restore_thread(self.thread_state)
        self.argman.emit_cleanup()


class PyCallWrapper(object):
    def __init__(self, context, module, func, fndesc, call_helper,
                 release_gil):
        self.context = context
        self.module = module
        self.func = func
        self.fndesc = fndesc
        self.release_gil = release_gil

    def build(self):
        wrapname = "wrapper.%s" % self.func.name

        # This is the signature of PyCFunctionWithKeywords
        # (see CPython's methodobject.h)
        pyobj = self.context.get_argument_type(types.pyobject)
        wrapty = Type.function(pyobj, [pyobj, pyobj, pyobj])
        wrapper = self.module.add_function(wrapty, name=wrapname)

        builder = Builder.new(wrapper.append_basic_block('entry'))

        # - `closure` will receive the `self` pointer stored in the
        #   PyCFunction object (see _dynfunc.c)
        # - `args` and `kws` will receive the tuple and dict objects
        #   of positional and keyword arguments, respectively.
        closure, args, kws = wrapper.args
        closure.name = 'py_closure'
        args.name = 'py_args'
        kws.name = 'py_kws'

        api = self.context.get_python_api(builder)
        self.build_wrapper(api, builder, closure, args, kws)

        return wrapper, api

    def build_wrapper(self, api, builder, closure, args, kws):
        nargs = len(self.fndesc.args)
        keywords = self.make_keywords(self.fndesc.args)
        fmt = self.make_const_string("O" * nargs)

        objs = [api.alloca_obj() for _ in range(nargs)]
        parseok = api.parse_tuple_and_keywords(args, kws, fmt, keywords, *objs)

        pred = builder.icmp(lc.ICMP_EQ, parseok, Constant.null(parseok.type))
        with cgutils.if_unlikely(builder, pred):
            builder.ret(api.get_null_object())

        # Block that returns after erroneous argument unboxing/cleanup
        endblk = cgutils.append_basic_block(builder, "arg.end")
        with cgutils.goto_block(builder, endblk):
            builder.ret(api.get_null_object())

        cleanup_manager = _ArgManager(self.context, builder, api, endblk, nargs)

        innerargs = []
        for obj, ty in zip(objs, self.fndesc.argtypes):
            val = cleanup_manager.add_arg(obj, ty)
            innerargs.append(val)

        if self.release_gil:
            cleanup_manager = _GilManager(builder, api, cleanup_manager)

        # The wrapped function doesn't take a full closure, only
        # the Environment object.
        env = self.context.get_env_from_closure(builder, closure)

        status, res = self.context.call_conv.call_function(
            builder, self.func, self.fndesc.restype, self.fndesc.argtypes,
            innerargs, env)
        # Do clean up
        cleanup_manager.emit_cleanup()

        # Determine return status
        with cgutils.if_likely(builder, status.is_ok):
            # Ok => return boxed Python value
            with cgutils.ifthen(builder, status.is_none):
                api.return_none()

            retval = api.from_native_return(res, self._simplified_return_type())
            builder.ret(retval)

        with cgutils.ifthen(builder, builder.not_(status.is_python_exc)):
            # User exception raised
            self.make_exception_switch(api, builder, status)

        # Error out
        builder.ret(api.get_null_object())

    def make_exception_switch(self, api, builder, status):
        """
        Handle user exceptions.  Unserialize the exception info and raise it.
        """
        code = status.code
        # Handle user exceptions
        with cgutils.ifthen(builder, status.is_user_exc):
            exc = api.unserialize(status.excinfoptr)
            with cgutils.if_likely(builder,
                                   cgutils.is_not_null(builder, exc)):
                api.raise_object(exc)  # steals ref
            builder.ret(api.get_null_object())

        with cgutils.ifthen(builder, status.is_stop_iteration):
            api.err_set_none("PyExc_StopIteration")
            builder.ret(api.get_null_object())

        msg = "unknown error in native function: %s" % self.fndesc.mangled_name
        api.err_set_string("PyExc_SystemError", msg)

    def make_const_string(self, string):
        return self.context.insert_const_string(self.module, string)

    def make_keywords(self, kws):
        strings = []
        stringtype = Type.pointer(Type.int(8))
        for k in kws:
            strings.append(self.make_const_string(k))

        strings.append(Constant.null(stringtype))
        kwlist = Constant.array(stringtype, strings)
        kwlist = cgutils.global_constant(self.module, ".kwlist", kwlist)
        return Constant.bitcast(kwlist, Type.pointer(stringtype))

    def _simplified_return_type(self):
        """
        The NPM callconv has already converted simplified optional types.
        We can simply use the value type from it.
        """
        restype = self.fndesc.restype
        # Optional type
        if isinstance(restype, types.Optional):
            return restype.type
        else:
            return restype



