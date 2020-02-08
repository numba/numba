from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         AbstractTemplate,
                                         MacroTemplate, signature, Registry)
from numba import roc

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
    key = roc.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_local_id(ConcreteTemplate):
    key = roc.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_group_id(ConcreteTemplate):
    key = roc.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_num_groups(ConcreteTemplate):
    key = roc.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_work_dim(ConcreteTemplate):
    key = roc.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Hsa_get_global_size(ConcreteTemplate):
    key = roc.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_get_local_size(ConcreteTemplate):
    key = roc.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Hsa_barrier(ConcreteTemplate):
    key = roc.barrier
    cases = [signature(types.void, types.uint32),
             signature(types.void)]


@intrinsic
class Hsa_mem_fence(ConcreteTemplate):
    key = roc.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Hsa_wavebarrier(ConcreteTemplate):
    key = roc.wavebarrier
    cases = [signature(types.void)]


@intrinsic
class Hsa_activelanepermute_wavewidth(ConcreteTemplate):
    key = roc.activelanepermute_wavewidth
    # parameter: src, laneid, identity, useidentity
    cases = [signature(ty, ty, types.uint32, ty, types.bool_)
             for ty in (types.integer_domain|types.real_domain)]


class _Hsa_ds_permuting(ConcreteTemplate):
    # parameter: index, source
    cases = [signature(types.int32, types.int32, types.int32),
             signature(types.int32, types.int64, types.int32),
             signature(types.float32, types.int32, types.float32),
             signature(types.float32, types.int64, types.float32)]
    unsafe_casting = False


@intrinsic
class Hsa_ds_permute(_Hsa_ds_permuting):
    key = roc.ds_permute


@intrinsic
class Hsa_ds_bpermute(_Hsa_ds_permuting):
    key = roc.ds_bpermute


# hsa.shared submodule -------------------------------------------------------

class Hsa_shared_array(MacroTemplate):
    key = roc.shared.array


@intrinsic_attr
class HsaSharedTemplate(AttributeTemplate):
    key = types.Module(roc.shared)

    def resolve_array(self, mod):
        return types.Macro(Hsa_shared_array)


# hsa.atomic submodule -------------------------------------------------------

@intrinsic
class Hsa_atomic_add(AbstractTemplate):
    key = roc.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class HsaAtomicTemplate(AttributeTemplate):
    key = types.Module(roc.atomic)

    def resolve_add(self, mod):
        return types.Function(Hsa_atomic_add)


# hsa module -----------------------------------------------------------------

@intrinsic_attr
class HsaModuleTemplate(AttributeTemplate):
    key = types.Module(roc)

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

    def resolve_ds_permute(self, mod):
        return types.Function(Hsa_ds_permute)

    def resolve_ds_bpermute(self, mod):
        return types.Function(Hsa_ds_bpermute)

    def resolve_shared(self, mod):
        return types.Module(roc.shared)

    def resolve_atomic(self, mod):
        return types.Module(roc.atomic)


# intrinsic

intrinsic_global(roc, types.Module(roc))
