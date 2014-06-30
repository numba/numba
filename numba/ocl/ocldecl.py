from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, MacroTemplate,
                                    signature, Registry)
from numba import ocl


registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global


@intrinsic
class Ocl_get_global_id(ConcreteTemplate):
    key = ocl.get_global_id
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Ocl_get_local_id(ConcreteTemplate):
    key = ocl.get_local_id
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Ocl_get_global_size(ConcreteTemplate):
    key = ocl.get_global_size
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Ocl_get_local_size(ConcreteTemplate):
    key = ocl.get_local_size
    cases = [signature(types.uintp, types.uint32)]


@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(ocl)

    def resolve_get_global_id(self, mod):
        return types.Function(Ocl_get_global_id)

    def resolve_get_local_id(self, mod):
        return types.Function(Ocl_get_local_id)

    def resolve_get_global_size(self, mod):
        return types.Function(Ocl_get_global_size)

    def resolve_get_local_size(self, mod):
        return types.Function(Ocl_get_local_size)


intrinsic_global(ocl, types.Module(ocl))
