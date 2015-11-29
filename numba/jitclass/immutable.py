from __future__ import absolute_import, print_function
from numba import types
from numba.typing import templates
from numba.datamodel import default_manager, models
from numba.targets import imputils
from numba import cgutils
from .base import ClassBuilder


class ImmInstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = list(fe_typ.struct.items())
        super(ImmInstanceModel, self).__init__(dmm, fe_typ, members)


class ImmRefModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        self._pointee_model = dmm.lookup(fe_type.instance_type)
        self._pointee_be_type = self._pointee_model.get_data_type()
        be_type = self._pointee_be_type.as_pointer()
        super(ImmRefModel, self).__init__(dmm, fe_type, be_type)


default_manager.register(types.ImmutableClassInstanceType, ImmInstanceModel)
default_manager.register(types.ImmutableClassType, models.OpaqueModel)
default_manager.register(types.ImmutableClassRefType, ImmRefModel)


class ImmutableClassBuilder(ClassBuilder):
    instance_type_class = types.ImmutableClassInstanceType

    def implement_frontend(self, instance_type):
        self.implement_attribute_typing(instance_type)
        self.register_byref_attribute_typing(instance_type)

    def register_byref_attribute_typing(self, instance_type):
        class RefStructAttribute(templates.AttributeTemplate):
            key = instance_type.get_reference_type()

            def __init__(self, context, instance):
                self.instance = instance.instance_type
                super(RefStructAttribute, self).__init__(context)

            def generic_resolve(self, instance, attr):
                structdct = instance.instance_type.struct
                if attr in structdct:
                    return structdct[attr]

        refattrspec = RefStructAttribute(self.typer,
                                         instance_type.get_reference_type())
        self.typer.insert_attributes(refattrspec)

    def implement_constructor(self, registry, instance_type, ctor_nargs):
        @registry.register
        @imputils.implement(self.class_type, *([types.Any] * ctor_nargs))
        def ctor_impl(context, builder, sig, args):
            # Allocate the instance
            inst_typ = sig.return_type

            inst_struct_typ = cgutils.create_struct_proxy(inst_typ)
            inst_struct = inst_struct_typ(context, builder)

            # Call the __init__
            # TODO: extract the following into a common util
            init_sig = (sig.return_type.get_reference_type(),) + sig.args

            init = self.jitmethods['__init__']
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

    def implement_attribute(self, registry, instance_type, attr):
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


defined_types = set()
