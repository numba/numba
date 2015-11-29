import types as pytypes
from numba import types
from numba.targets.registry import CPUTarget
from numba import njit
from numba.typing import templates
from numba.datamodel import default_manager, models
from numba.targets import imputils
from numba import cgutils
from llvmlite import ir as llvmir


def jitclass(spec):
    if not callable(spec):
        specfn = lambda *args, **kwargs: spec
    else:
        specfn = spec

    def wrap(cls):
        register_class_type(cls, specfn)
        return cls

    return wrap


class ClassModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        cls_data_ty = types.ClassDataType(fe_typ)
        # MemInfoPointer uses the `dtype` attribute to traverse for nested
        # NRT MemInfo.  Since we handle nested NRT MemInfo ourselves,
        # we will replace provide MemInfoPointer with an opaque type
        # so that it does not raise exception for nested meminfo.
        dtype = types.Opaque('Opaque.' + str(cls_data_ty))
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
            ('data', types.CPointer(cls_data_ty)),
        ]
        super(ClassModel, self).__init__(dmm, fe_typ, members)


class ClassDataModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        clsty = fe_typ.class_type
        members = list(clsty.struct.items())
        super(ClassDataModel, self).__init__(dmm, fe_typ, members)


default_manager.register(types.ClassInstanceType, ClassModel)
default_manager.register(types.ClassDataType, ClassDataModel)
default_manager.register(types.ClassType, models.OpaqueModel)


def register_class_type(cls, specfn):
    clsdct = cls.__dict__
    methods = dict((k, v) for k, v in clsdct.items()
                   if isinstance(v, pytypes.FunctionType))
    # classname = cls.__name__

    class_type = types.ClassType(cls)

    jitmethods = {}
    for k, v in methods.items():
        jitmethods[k] = njit(v)

    # Register resolution of the class object
    typer = CPUTarget.typing_context
    typer.insert_global(cls, class_type)

    def defer_frontend(instance_type):
        # Register typing of the class attributes

        class ClassAttribute(templates.AttributeTemplate):
            key = instance_type

            def __init__(self, context, instance):
                self.instance = instance
                super(ClassAttribute, self).__init__(context)

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

        attrspec = ClassAttribute(typer, instance_type)
        typer.insert_attributes(attrspec)

    ### Backend ###
    backend = CPUTarget.target_context

    registry = imputils.Registry()

    # Add constructor
    ctor_nargs = len(jitmethods['__init__']._pysig.parameters) - 1

    def defer_backend(instance_type):
        def imp_dtor(context, module):
            dtor_ftype = llvmir.FunctionType(llvmir.VoidType(),
                                             [context.get_value_type(
                                                 types.voidptr),
                                              context.get_value_type(
                                                  types.voidptr)])

            fname = "_Dtor.{0}".format(instance_type.name)
            dtor_fn = module.get_or_insert_function(dtor_ftype,
                                                    name=fname)
            if dtor_fn.is_declaration:
                # Define
                builder = llvmir.IRBuilder(dtor_fn.append_basic_block())

                alloc_fe_type = instance_type.get_data_type()
                alloc_type = context.get_value_type(alloc_fe_type)

                data_struct = cgutils.create_struct_proxy(alloc_fe_type)

                ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
                data = data_struct(context, builder, ref=ptr)

                context.nrt_decref(builder, alloc_fe_type, data._getvalue())

                builder.ret_void()

            return dtor_fn

        @registry.register
        @imputils.implement(class_type, *([types.Any] * ctor_nargs))
        def ctor_impl(context, builder, sig, args):
            # Allocate the instance
            inst_typ = sig.return_type
            alloc_type = context.get_value_type(inst_typ.get_data_type())
            alloc_size = context.get_abi_sizeof(alloc_type)

            meminfo = context.nrt_meminfo_alloc_dtor(
                builder,
                context.get_constant(types.uintp, alloc_size),
                imp_dtor(context, builder.module),
            )
            data_pointer = context.nrt_meminfo_data(builder, meminfo)
            data_pointer = builder.bitcast(data_pointer,
                                           alloc_type.as_pointer())

            # Nullify all data
            builder.store(cgutils.get_null_value(alloc_type),
                          data_pointer)

            inst_struct_typ = cgutils.create_struct_proxy(inst_typ)
            inst_struct = inst_struct_typ(context, builder)
            inst_struct.meminfo = meminfo
            inst_struct.data = data_pointer

            # Call the __init__
            # TODO: extract the following into a common util
            init_sig = (sig.return_type,) + sig.args

            init = jitmethods['__init__']
            init.compile(init_sig)
            cres = init._compileinfos[init_sig]
            realargs = [inst_struct._getvalue()] + list(args)
            context.call_internal(builder, cres.fndesc, types.void(*init_sig),
                                  realargs)

            # Prepare reutrn value
            ret = inst_struct._getvalue()

            # Add function to link
            codegen = context.codegen()
            codegen.add_linking_library(cres.library)

            return imputils.impl_ret_new_ref(context, builder, inst_typ, ret)

        # Add attributes

        def make_attr(attr):
            @registry.register_attr
            @imputils.impl_attribute(instance_type, attr)
            def imp(context, builder, typ, value):
                inst_struct = cgutils.create_struct_proxy(typ)
                inst = inst_struct(context, builder, value=value)
                data_pointer = inst.data
                data_struct = cgutils.create_struct_proxy(typ.get_data_type())
                data = data_struct(context, builder, ref=data_pointer)
                return imputils.impl_ret_borrowed(context, builder,
                                                  typ.struct[attr],
                                                  getattr(data, attr))

        for attr in instance_type.struct:
            make_attr(attr)

        # Add methods
        def make_method(attr):
            nargs = len(jitmethods[attr]._pysig.parameters)

            @registry.register
            @imputils.implement((instance_type, attr), *([types.Any] * nargs))
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

    # Register constructor
    class ConstructorTemplate(templates.AbstractTemplate):
        key = class_type

        def generic(self, args, kws):
            ctor = jitmethods['__init__']

            struct = specfn(*args, **kws)
            instance_type = types.ClassInstanceType(class_type=class_type,
                                                    struct=struct,
                                                    methods=methods)

            if instance_type not in defined_types:
                defined_types.add(instance_type)
                defer_frontend(instance_type)
                defer_backend(instance_type)

            boundargs = (instance_type,) + args
            template, args, kws = ctor.get_call_template(boundargs, kws)
            sig = template(self.context).apply(args, kws)
            out = templates.signature(sig.args[0], *sig.args[1:])
            return out

    typer.insert_function(ConstructorTemplate(typer))


defined_types = set()
