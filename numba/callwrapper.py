from __future__ import print_function, division, absolute_import

from llvm.core import Type, Builder, Constant
import llvm.core as lc

from numba import types, cgutils, errcode


class _ArgManager(object):
    """
    A utility class to handle argument unboxing and cleanup
    """
    def __init__(self, builder, api, nargs):
        self.builder = builder
        self.api = api
        self.arg_count = 0  # how many function arguments have been processed
        self.cleanups = []

        # set up switch for error processing of function arguments
        self.elseblk = cgutils.append_basic_block(self.builder, "arg.ok")
        with cgutils.goto_block(self.builder, self.elseblk):
            self.builder.ret(self.api.get_null_object())

        self.swtblk = cgutils.append_basic_block(self.builder, ".arg.err")
        with cgutils.goto_block(self.builder, self.swtblk):
            self.swt_val = cgutils.alloca_once(self.builder, Type.int(32))
            self.swt = self.builder.switch(self.builder.load(self.swt_val),
                                           self.elseblk, nargs)

        self.prev = self.elseblk

    def add_arg(self, obj, ty):
        """
        Unbox argument and emit code that handles any error during unboxing
        """
        # Unbox argument
        val, dtor = self.api.to_native_arg(self.builder.load(obj), ty)
        self.cleanups.append(dtor)
        # add to the switch each time through the loop
        # prev and cur are references to keep track of which block to branch to

        if self.arg_count == 0:
            bb = cgutils.append_basic_block(self.builder,
                                            "arg%d.err" % self.arg_count)
            self.cur = bb
            self.swt.add_case(Constant.int(Type.int(32), self.arg_count), bb)
        else:
            # keep a reference to the previous arg.error block
            self.prev = self.cur
            bb = cgutils.append_basic_block(self.builder,
                                            "arg%d.error" % self.arg_count)
            self.cur = bb
            self.swt.add_case(Constant.int(Type.int(32), self.arg_count), bb)

        # write the error block
        with cgutils.goto_block(self.builder, self.cur):
            dtor()
            self.builder.branch(self.prev)

        # store arg count into value to switch on if there is an error
        self.builder.store(Constant.int(Type.int(32), self.arg_count),
                           self.swt_val)

        # check for Python C-API Error
        error_check = self.api.err_occurred()
        err_happened = self.builder.icmp(lc.ICMP_NE, error_check,
                                         self.api.get_null_object())
        # if error occurs -- clean up -- goto switch block
        with cgutils.if_unlikely(self.builder, err_happened):
            self.builder.branch(self.swtblk)

        self.arg_count += 1
        return val

    def emit_cleanup(self):
        """Emit the cleanup code after we are done with the arguments
        """
        for dtor in self.cleanups:
            dtor()


class PyCallWrapper(object):
    def __init__(self, context, module, func, fndesc, exceptions):
        self.context = context
        self.module = module
        self.func = func
        self.fndesc = fndesc
        self.exceptions = exceptions

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

        wrapper.verify()
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

        argman = _ArgManager(builder, api, nargs)

        innerargs = []
        for obj, ty in zip(objs, self.fndesc.argtypes):
            val = argman.add_arg(obj, ty)
            innerargs.append(val)

        # The wrapped function doesn't take a full closure, only
        # the Environment object.
        env = self.context.get_env_from_closure(builder, closure)

        status, res = self.context.call_function(builder, self.func,
                                                 self.fndesc.restype,
                                                 self.fndesc.argtypes,
                                                 innerargs, env)
        # Do clean up
        argman.emit_cleanup()

        # Determine return status
        with cgutils.if_likely(builder, status.ok):
            with cgutils.ifthen(builder, status.none):
                api.return_none()

            retval = api.from_native_return(res, self.fndesc.restype)
            builder.ret(retval)

        with cgutils.ifthen(builder, builder.not_(status.exc)):
            # !ok && !exc
            # User exception raised
            self.make_exception_switch(api, builder, status.code)

        # !ok && exc
        builder.ret(api.get_null_object())

    def make_exception_switch(self, api, builder, code):
        """Handle user defined exceptions.
        Build a switch to check which exception class was raised.
        """
        nexc = len(self.exceptions)
        elseblk = cgutils.append_basic_block(builder, ".invalid.user.exception")
        swt = builder.switch(code, elseblk, n=nexc)
        for num, exc in self.exceptions.items():
            bb = cgutils.append_basic_block(builder,
                                            ".user.exception.%d" % num)
            swt.add_case(Constant.int(code.type, num), bb)
            builder.position_at_end(bb)
            api.raise_exception(exc, exc)
            builder.ret(api.get_null_object())

        builder.position_at_end(elseblk)

        # Handle native error
        elseblk = cgutils.append_basic_block(builder, ".invalid.native.error")
        swt = builder.switch(code, elseblk, n=len(errcode.error_names))

        msgfmt = "{error} in native function: {fname}"
        for errnum, errname in errcode.error_names.items():
            bb = cgutils.append_basic_block(builder,
                                            ".native.error.%d" % errnum)
            swt.add_case(Constant.int(code.type, errnum), bb)
            builder.position_at_end(bb)

            api.raise_native_error(msgfmt.format(error=errname,
                                                 fname=self.fndesc.mangled_name))
            builder.ret(api.get_null_object())

        builder.position_at_end(elseblk)
        msg = "unknown error in native function: %s" % self.fndesc.mangled_name
        api.raise_native_error(msg)

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
