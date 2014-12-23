from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)
from numba import hsa


registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global


@intrinsic
class Hsa_get_global_id(ConcreteTemplate):
    key = hsa.get_global_id
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Hsa_get_local_id(ConcreteTemplate):
    key = hsa.get_local_id
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Hsa_get_global_size(ConcreteTemplate):
    key = hsa.get_global_size
    cases = [signature(types.uintp, types.uint32)]


@intrinsic
class Hsa_get_local_size(ConcreteTemplate):
    key = hsa.get_local_size
    cases = [signature(types.uintp, types.uint32)]


@intrinsic_attr
class HsaModuleTemplate(AttributeTemplate):
    key = types.Module(hsa)

    def resolve_get_global_id(self, mod):
        return types.Function(Hsa_get_global_id)

    def resolve_get_local_id(self, mod):
        return types.Function(Hsa_get_local_id)

    def resolve_get_global_size(self, mod):
        return types.Function(Hsa_get_global_size)

    def resolve_get_local_size(self, mod):
        return types.Function(Hsa_get_local_size)


intrinsic_global(hsa, types.Module(hsa))
