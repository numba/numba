import types as pytypes
from numba import types
from numba.targets.registry import CPUTarget
from numba import njit
from numba.typing import templates
from numba.datamodel import default_manager, models
from numba.targets import imputils
from numba import cgutils
from llvmlite import ir as llvmir


def jitstruct(spec):
    if not callable(spec):
        specfn = lambda *args, **kwargs: spec
    else:
        specfn = spec

    def wrap(cls):
        register_struct_type(cls, specfn)
        return cls

    return wrap


class StructInstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = list(fe_typ.struct.items())
        super(StructInstanceModel, self).__init__(dmm, fe_typ, members)


class StructRefModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        self._pointee_model = dmm.lookup(fe_type.instance_type)
        self._pointee_be_type = self._pointee_model.get_data_type()
        be_type = self._pointee_be_type.as_pointer()
        super(StructRefModel, self).__init__(dmm, fe_type, be_type)


default_manager.register(types.StructInstanceType, StructInstanceModel)
default_manager.register(types.StructClassType, models.OpaqueModel)
default_manager.register(types.StructRefType, StructRefModel)


def register_struct_type(cls, specfn):
    clsdct = cls.__dict__
    methods = dict((k, v) for k, v in clsdct.items()
                   if isinstance(v, pytypes.FunctionType))
    # classname = cls.__name__


    class_type = types.StructClassType(cls)

    jitmethods = {}
    for k, v in methods.items():
        jitmethods[k] = njit(v)

    # Register resolution of the class object
    typer = CPUTarget.typing_context
    typer.insert_global(cls, class_type)

    def defer_frontend(instance_type):

        # Register typing of the class attributes

        class StructAttribute(templates.AttributeTemplate):
            key = instance_type

            def __init__(self, context, instance):
                self.instance = instance
                super(StructAttribute, self).__init__(context)

            def generic_resolve(self, instance, attr):

                if attr in instance.struct:
                    return instance.struct[attr]

                elif attr in instance.methods:
                    meth = jitmethods[attr]

                    class MethodTemplate(templates.AbstractTemplate):
                        key = (instance_type, attr)

                        def generic(self, args, kws):
                            args = (instance,) + tuple(args)
                            template, args, kws = meth.get_call_template(args,
                                                                         kws)
                            sig = template(self.context).apply(args, kws)
                            sig = templates.signature(sig.return_type,
                                                      *sig.args[1:],
                                                      recvr=sig.args[0])
                            return sig

                    return types.BoundFunction(MethodTemplate, instance)

        attrspec = StructAttribute(typer, instance_type)
        typer.insert_attributes(attrspec)

        class RefStructAttribute(templates.AttributeTemplate):
            key = instance_type.get_reference_type()

            def __init__(self, context, instance):
                self.instance = instance.instance_type
                super(RefStructAttribute, self).__init__(context)

            def generic_resolve(self, instance, attr):
                structdct = instance.instance_type.struct
                if attr in structdct:
                    return structdct[attr]
                    #
                    # elif attr in instance.methods:
                    #     meth = jitmethods[attr]
                    #
                    #     class MethodTemplate(templates.AbstractTemplate):
                    #         key = (instance_type, attr)
                    #
                    #         def generic(self, args, kws):
                    #             args = (instance,) + tuple(args)
                    #             template, args, kws = meth.get_call_template(args, kws)
                    #             sig = template(self.context).apply(args, kws)
                    #             sig = templates.signature(sig.return_type,
                    #                                       *sig.args[1:],
                    #                                       recvr=sig.args[0])
                    #             return sig
                    #
                    #     return types.BoundFunction(MethodTemplate, instance)

        refattrspec = RefStructAttribute(typer,
                                         instance_type.get_reference_type())
        typer.insert_attributes(refattrspec)

    ### Backend ###
    backend = CPUTarget.target_context

    registry = imputils.Registry()

    # Add constructor
    ctor_nargs = len(jitmethods['__init__']._pysig.parameters) - 1

    def defer_backend(instance_type):
        # Add attributes

        def make_attr(attr):
            @registry.register_attr
            @imputils.impl_attribute(instance_type, attr)
            def imp(context, builder, typ, value):
                inst_struct = cgutils.create_struct_proxy(typ)
                inst = inst_struct(context, builder, value=value)
                return imputils.impl_ret_borrowed(context, builder,
                                                  typ.struct[attr],
                                                  getattr(inst, attr))

            @registry.register_attr
            @imputils.impl_attribute(instance_type.get_reference_type(), attr)
            def imp(context, builder, typ, value):
                inst_struct = cgutils.create_struct_proxy(typ)
                inst = inst_struct(context, builder, ref=value)
                return imputils.impl_ret_borrowed(context, builder,
                                                  typ.struct[attr],
                                                  getattr(inst, attr))

        for attr in instance_type.struct:
            make_attr(attr)

        # Add methods
        def make_method(attr):
            nargs = len(jitmethods[attr]._pysig.parameters)
            self_type = (instance_type
                         if attr != '__init__'
                         else instance_type.get_reference_type())

            @registry.register
            @imputils.implement((self_type, attr), *([types.Any] * nargs))
            def imp(context, builder, sig, args):
                method = jitmethods[attr]
                method.compile(sig)
                cres = method._compileinfos[sig.args]
                out = context.call_internal(builder, cres.fndesc, sig, args)
                return imputils.impl_ret_new_ref(context, builder,
                                                 sig.return_type, out)

        for meth in instance_type.methods:
            make_method(meth)

        backend.insert_func_defn(registry.functions)
        backend.insert_attr_defn(registry.attributes)

    @registry.register
    @imputils.implement(class_type, *([types.Any] * ctor_nargs))
    def ctor_impl(context, builder, sig, args):
        # Allocate the instance
        inst_typ = sig.return_type

        inst_struct_typ = cgutils.create_struct_proxy(inst_typ)
        inst_struct = inst_struct_typ(context, builder)

        # Call the __init__
        # TODO: extract the following into a common util
        init_sig = (sig.return_type.get_reference_type(),) + sig.args

        init = jitmethods['__init__']
        init.compile(init_sig)
        cres = init._compileinfos[init_sig]
        realargs = [inst_struct._getpointer()] + list(args)
        context.call_internal(builder, cres.fndesc, types.void(*init_sig),
                              realargs)

        # Prepare return value
        ret = inst_struct._getvalue()

        # Add function to link
        codegen = context.codegen()
        codegen.add_linking_library(cres.library)

        return imputils.impl_ret_new_ref(context, builder, inst_typ, ret)

    # ------------
    # Constructor

    # Register constructor
    class ConstructorTemplate(templates.AbstractTemplate):
        key = class_type

        def generic(self, args, kws):
            ctor = jitmethods['__init__']
            struct = specfn(*args, **kws)
            instance_type = types.StructInstanceType(class_type=class_type,
                                                     struct=struct,
                                                     methods=methods)

            if instance_type not in defined_types:
                defined_types.add(instance_type)
                defer_frontend(instance_type)
                defer_backend(instance_type)

            boundargs = (instance_type.get_reference_type(),) + args
            template, args, kws = ctor.get_call_template(boundargs, kws)
            sig = template(self.context).apply(args, kws)
            out = templates.signature(sig.args[0].instance_type, *sig.args[1:])
            return out

    typer.insert_function(ConstructorTemplate(typer))


defined_types = set()
