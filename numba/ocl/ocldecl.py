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
    key = Ocl.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_id(ConcreteTemplate):
    key = Ocl.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_group_id(ConcreteTemplate):
    key = Ocl.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_num_groups(ConcreteTemplate):
    key = Ocl.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_work_dim(ConcreteTemplate):
    key = Ocl.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Ocl_get_global_size(ConcreteTemplate):
    key = Ocl.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_size(ConcreteTemplate):
    key = Ocl.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_barrier(ConcreteTemplate):
    key = Ocl.barrier
    cases = [signature(types.void, types.uint32),
             signature(types.void)]


@intrinsic
class Ocl_mem_fence(ConcreteTemplate):
    key = Ocl.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Ocl_wavebarrier(ConcreteTemplate):
    key = Ocl.wavebarrier
    cases = [signature(types.void)]

@intrinsic
class Ocl_activelanepermute_wavewidth(ConcreteTemplate):
    key = Ocl.activelanepermute_wavewidth
    # parameter: src, laneid, identity, useidentity
    cases = [signature(ty, ty, types.uint32, ty, types.bool_)
             for ty in (types.integer_domain|types.real_domain)]

# Ocl.shared submodule -------------------------------------------------------

class Ocl_shared_array(MacroTemplate):
    key = Ocl.shared.array


@intrinsic_attr
class OclSharedTemplate(AttributeTemplate):
    key = types.Module(Ocl.shared)

    def resolve_array(self, mod):
        return types.Macro(Ocl_shared_array)


# Ocl.atomic submodule -------------------------------------------------------

@intrinsic
class Ocl_atomic_add(AbstractTemplate):
    key = Ocl.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class OclAtomicTemplate(AttributeTemplate):
    key = types.Module(Ocl.atomic)

    def resolve_add(self, mod):
        return types.Function(Ocl_atomic_add)


# Ocl module -----------------------------------------------------------------

@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(Ocl)

    def resolve_get_global_id(self, mod):
        return types.Function(Ocl_get_global_id)

    def resolve_get_local_id(self, mod):
        return types.Function(Ocl_get_local_id)

    def resolve_get_global_size(self, mod):
        return types.Function(Ocl_get_global_size)

    def resolve_get_local_size(self, mod):
        return types.Function(Ocl_get_local_size)

    def resolve_get_num_groups(self, mod):
        return types.Function(Ocl_get_num_groups)

    def resolve_get_work_dim(self, mod):
        return types.Function(Ocl_get_work_dim)

    def resolve_get_group_id(self, mod):
        return types.Function(Ocl_get_group_id)

    def resolve_barrier(self, mod):
        return types.Function(Ocl_barrier)

    def resolve_mem_fence(self, mod):
        return types.Function(Ocl_mem_fence)

    def resolve_sub_group_barrier(self, mod):
        return types.Function(Ocl_sub_group_barrier)

    def resolve_shared(self, mod):
        return types.Module(Ocl.shared)

    def resolve_atomic(self, mod):
        return types.Module(Ocl.atomic)


# intrinsic

intrinsic_global(Ocl, types.Module(Ocl))
