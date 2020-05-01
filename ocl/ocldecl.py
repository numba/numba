from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.npydecl import register_number_classes
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, MacroTemplate,
                                    signature, Registry)
from numba import dppy

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
#intrinsic_global = registry.register_global

#register_number_classes(intrinsic_global)

@intrinsic
class Ocl_get_global_id(ConcreteTemplate):
    key = dppy.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_id(ConcreteTemplate):
    key = dppy.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_group_id(ConcreteTemplate):
    key = dppy.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_num_groups(ConcreteTemplate):
    key = dppy.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_work_dim(ConcreteTemplate):
    key = dppy.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Ocl_get_global_size(ConcreteTemplate):
    key = dppy.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_size(ConcreteTemplate):
    key = dppy.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_barrier(ConcreteTemplate):
    key = dppy.barrier
    cases = [signature(types.intp, types.int32),
             signature(types.uintp, types.uint32)]


@intrinsic
class Ocl_mem_fence(ConcreteTemplate):
    key = dppy.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Ocl_sub_group_barrier(ConcreteTemplate):
    key = dppy.sub_group_barrier
    cases = [signature(types.void)]


# ocl.atomic submodule -------------------------------------------------------

@intrinsic
class Ocl_atomic_add(AbstractTemplate):
    key = dppy.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)

@intrinsic
class Ocl_atomic_sub(AbstractTemplate):
    key = dppy.atomic.sub

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class OclAtomicTemplate(AttributeTemplate):
    key = types.Module(dppy.atomic)

    def resolve_add(self, mod):
        return types.Function(Ocl_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Ocl_atomic_sub)


# ocl.shared submodule -------------------------------------------------------

#class Ocl_shared_array(MacroTemplate):
#    key = ocl.shared.array


@intrinsic_attr
#class OclSharedTemplate(AttributeTemplate):
#    key = types.Module(ocl.shared)

#    def resolve_array(self, mod):
#        return types.Macro(Ocl_shared_array)

# OpenCL module --------------------------------------------------------------

@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(dppy)

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
        return types.Module(dppy.atomic)

#    def resolve_shared(self, mod):
#        return types.Module(ocl.shared)

# intrinsic

#intrinsic_global(dppy, types.Module(dppy))
