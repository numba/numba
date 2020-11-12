from __future__ import print_function, division, absolute_import
from numba import types
from numba.core.typing.npydecl import register_number_classes
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                        AbstractTemplate, MacroTemplate,
                                        signature, Registry)
from numba import dppl

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
#intrinsic_global = registry.register_global

#register_number_classes(intrinsic_global)

@intrinsic
class Ocl_get_global_id(ConcreteTemplate):
    key = dppl.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_id(ConcreteTemplate):
    key = dppl.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_group_id(ConcreteTemplate):
    key = dppl.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_num_groups(ConcreteTemplate):
    key = dppl.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_work_dim(ConcreteTemplate):
    key = dppl.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Ocl_get_global_size(ConcreteTemplate):
    key = dppl.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_size(ConcreteTemplate):
    key = dppl.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_barrier(ConcreteTemplate):
    key = dppl.barrier
    cases = [signature(types.void, types.uint32),
             signature(types.void)]


@intrinsic
class Ocl_mem_fence(ConcreteTemplate):
    key = dppl.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Ocl_sub_group_barrier(ConcreteTemplate):
    key = dppl.sub_group_barrier

    cases = [signature(types.void)]


# dppl.atomic submodule -------------------------------------------------------

@intrinsic
class Ocl_atomic_add(AbstractTemplate):
    key = dppl.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)

@intrinsic
class Ocl_atomic_sub(AbstractTemplate):
    key = dppl.atomic.sub

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class OclAtomicTemplate(AttributeTemplate):
    key = types.Module(dppl.atomic)

    def resolve_add(self, mod):
        return types.Function(Ocl_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Ocl_atomic_sub)


# dppl.local submodule -------------------------------------------------------

class Ocl_local_alloc(MacroTemplate):
    key = dppl.local.static_alloc


@intrinsic_attr
class OclLocalTemplate(AttributeTemplate):
    key = types.Module(dppl.local)

    def resolve_static_alloc(self, mod):
        return types.Macro(Ocl_local_alloc)


# OpenCL module --------------------------------------------------------------

@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(dppl)

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

    def resolve_atomic(self, mod):
        return types.Module(dppl.atomic)

    def resolve_local(self, mod):
        return types.Module(dppl.local)

# intrinsic

#intrinsic_global(dppl, types.Module(dppl))
