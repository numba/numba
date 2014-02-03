from __future__ import print_function, division, absolute_import
from llvm.core import Type, Builder, Constant
import llvm.core as lc
from numba import types, cgutils


class PyCallWrapper(object):
    def __init__(self, context, module, func, fndesc):
        self.context = context
        self.module = module
        self.func = func
        self.fndesc = fndesc

    def build(self):
        wrapname = "wrapper.%s" % self.func.name

        pyobj = self.context.get_argument_type(types.pyobject)
        fnty = Type.function(pyobj, [pyobj, pyobj, pyobj])
        wrapper = self.module.add_function(fnty, name=wrapname)

        builder = Builder.new(wrapper.append_basic_block('entry'))

        _, args, kws = wrapper.args

        api = self.context.get_python_api(builder)
        self.build_wrapper(api, builder, args, kws)

        wrapper.verify()
        return wrapper, api

    def build_wrapper(self, api, builder, args, kws):
        nargs = len(self.fndesc.args)
        keywords = self.make_keywords(self.fndesc.args)
        fmt = self.make_const_string("O" * nargs)

        objs = [api.alloca_obj() for _ in range(nargs)]
        parseok = api.parse_tuple_and_keywords(args, kws, fmt, keywords, *objs)

        pred = builder.icmp(lc.ICMP_EQ, parseok, Constant.null(parseok.type))
        with cgutils.if_unlikely(builder, pred):
            builder.ret(api.get_null_object())

        innerargs = []
        for obj, ty in zip(objs, self.fndesc.argtypes):
            #api.context.debug_print(builder, "%s -> %s" % (obj, ty))
            #api.print_object(builder.load(obj))
            val = api.to_native_arg(builder.load(obj), ty)
            innerargs.append(val)

        status, res = self.context.call_function(builder, self.func,
                                                 self.fndesc.argtypes,
                                                 innerargs)

        with cgutils.if_likely(builder, status.ok):
            with cgutils.ifthen(builder, status.none):
                api.return_none()

            retval = api.from_native_return(res, self.fndesc.restype)
            builder.ret(retval)

        with cgutils.ifthen(builder, builder.not_(status.exc)):
            # User exception raised
            # TODO we will just raise a RuntimeError for now.
            api.raise_native_error("error in native function: %s" %
                                   self.fndesc.mangled_name)

        builder.ret(api.get_null_object())

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

