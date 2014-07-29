from __future__ import print_function, division, absolute_import

from llvm.core import Type, Builder, Constant
import llvm.core as lc

from numba import types, cgutils


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

        # set up switch for error processing of function arguments
        with cgutils.goto_block(builder, builder.basic_block): 
            elseblk = cgutils.append_basic_block(builder, ".return.switch.elseblk.on.error")
            swt_val = cgutils.alloca_once(builder, Type.int(32))
            swtblk = cgutils.append_basic_block(builder, ".switch.on.error")
            with cgutils.goto_block(builder, swtblk): 
                swt = builder.switch(builder.load(swt_val), elseblk, nargs)

        innerargs = []
        cleanups = []
        arg_count = 0  # how many function arguments have been processed
        for obj, ty in zip(objs, self.fndesc.argtypes):
            #api.context.debug_print(builder, "%s -> %s" % (obj, ty))
            #api.print_object(builder.load(obj))
            val, dtor = api.to_native_arg(builder.load(obj), ty)
            
            # add to the switch each time through the loop 
            # prev and cur are references to keep track of which block to branch to
            if arg_count == 0:
                with cgutils.goto_block(builder, elseblk): 
                    # build the elseblk of the switch statement 
                    # code should never reach here -- keeping llvm ir happy
                    builder.position_at_end(elseblk)
                    builder.ret(api.get_null_object())
                bb = cgutils.append_basic_block(builder, "switch.arg.error.%d" % arg_count)
                cur = bb
                swt.add_case(Constant.int(Type.int(32), arg_count), bb)
            else:
                prev = cur  # keep a reference to the previous arg.error block
                bb = cgutils.append_basic_block(builder, "switch.arg.error.%d" % arg_count)
                cur = bb
                swt.add_case(Constant.int(Type.int(32), arg_count), bb)
            
            # write the error block  
            with cgutils.goto_block(builder, cur):
                dtor()
                if arg_count == 0:  # every switch block should fall through to here
                    builder.ret(api.get_null_object())
                else:
                    builder.branch(prev)
            # store arg count into value to switch on if there is an error
            builder.store(Constant.int(Type.int(32), arg_count), swt_val)
            
            # check for Python C-API Error
            error_check = api.err_occurred()
            err_happened = builder.icmp(lc.ICMP_NE, error_check, api.get_null_object())
            # if error occurs -- clean up -- goto switch block
            with cgutils.if_unlikely(builder, err_happened):
                builder.branch(swtblk)

            innerargs.append(val)
            cleanups.append(dtor)
            arg_count += 1

        # The wrapped function doesn't take a full closure, only
        # the Environment object.
        env = self.context.get_env_from_closure(builder, closure)

        status, res = self.context.call_function(builder, self.func,
                                                 self.fndesc.restype,
                                                 self.fndesc.argtypes,
                                                 innerargs, env)
        # Do clean up
        for dtor in cleanups:
            dtor()
 
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
        builder.basic_block.function.viewCFG()


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
        msg = "error in native function: %s" % self.fndesc.mangled_name
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

        gv = self.module.add_global_variable(kwlist.type, name=".kwlist")
        gv.global_constant = True
        gv.initializer = kwlist
        gv.linkage = lc.LINKAGE_INTERNAL

        return Constant.bitcast(gv, Type.pointer(stringtype))

