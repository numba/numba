from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate,
                                    MacroTemplate, signature, Registry)
from numba import hsa

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global


# =============================== NOTE ===============================
# Even though the following functions return size_t in the OpenCL standard,
# It should be rare (and unrealistic) to have 2**63 number of work items.
# We are choosing to use intp (signed 64-bit in large model) due to potential
# loss of precision in coerce(intp, uintp) that results in double.


@intrinsic
class Hsa_get_global_id(ConcreteTemplate):
    key = hsa.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_local_id(ConcreteTemplate):
    key = hsa.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_group_id(ConcreteTemplate):
    key = hsa.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_num_groups(ConcreteTemplate):
    key = hsa.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_work_dim(ConcreteTemplate):
    key = hsa.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Hsa_get_global_size(ConcreteTemplate):
    key = hsa.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_local_size(ConcreteTemplate):
    key = hsa.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_barrier(ConcreteTemplate):
    key = hsa.barrier
    cases = [signature(types.void, types.uint32),
             signature(types.void)]


@intrinsic
class Hsa_mem_fence(ConcreteTemplate):
    key = hsa.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Hsa_wavebarrier(ConcreteTemplate):
    key = hsa.wavebarrier
    cases = [signature(types.void)]

@intrinsic
class Hsa_activelanepermute_wavewidth(ConcreteTemplate):
    key = hsa.activelanepermute_wavewidth
    # parameter: src, laneid, identity, useidentity
    cases = [signature(ty, ty, types.uint32, ty, types.bool_)
             for ty in (types.integer_domain|types.real_domain)]

# hsa.shared submodule -------------------------------------------------------

class Hsa_shared_array(MacroTemplate):
    key = hsa.shared.array


@intrinsic_attr
class HsaSharedTemplate(AttributeTemplate):
    key = types.Module(hsa.shared)

    def resolve_array(self, mod):
        return types.Macro(Hsa_shared_array)


# hsa.atomic submodule -------------------------------------------------------

@intrinsic
class Hsa_atomic_add(AbstractTemplate):
    key = hsa.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class HsaAtomicTemplate(AttributeTemplate):
    key = types.Module(hsa.atomic)

    def resolve_add(self, mod):
        return types.Function(Hsa_atomic_add)


# hsa module -----------------------------------------------------------------

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

    def resolve_get_num_groups(self, mod):
        return types.Function(Hsa_get_num_groups)

    def resolve_get_work_dim(self, mod):
        return types.Function(Hsa_get_work_dim)

    def resolve_get_group_id(self, mod):
        return types.Function(Hsa_get_group_id)

    def resolve_barrier(self, mod):
        return types.Function(Hsa_barrier)

    def resolve_mem_fence(self, mod):
        return types.Function(Hsa_mem_fence)

    def resolve_wavebarrier(self, mod):
        return types.Function(Hsa_wavebarrier)

    def resolve_activelanepermute_wavewidth(self, mod):
        return types.Function(Hsa_activelanepermute_wavewidth)

    def resolve_shared(self, mod):
        return types.Module(hsa.shared)

    def resolve_atomic(self, mod):
        return types.Module(hsa.atomic)


# intrinsic

intrinsic_global(hsa, types.Module(hsa))
