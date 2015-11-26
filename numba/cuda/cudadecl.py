from __future__ import print_function, division, absolute_import
from numba import types
from numba.typing.npydecl import register_number_classes
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, MacroTemplate,
                                    signature, Registry)
from numba import cuda


registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global

register_number_classes(intrinsic_global)


class Cuda_grid(MacroTemplate):
    key = cuda.grid


class Cuda_gridsize(MacroTemplate):
    key = cuda.gridsize


class Cuda_threadIdx_x(MacroTemplate):
    key = cuda.threadIdx.x


class Cuda_threadIdx_y(MacroTemplate):
    key = cuda.threadIdx.y


class Cuda_threadIdx_z(MacroTemplate):
    key = cuda.threadIdx.z


class Cuda_blockIdx_x(MacroTemplate):
    key = cuda.blockIdx.x


class Cuda_blockIdx_y(MacroTemplate):
    key = cuda.blockIdx.y


class Cuda_blockIdx_z(MacroTemplate):
    key = cuda.blockIdx.z


class Cuda_blockDim_x(MacroTemplate):
    key = cuda.blockDim.x


class Cuda_blockDim_y(MacroTemplate):
    key = cuda.blockDim.y


class Cuda_blockDim_z(MacroTemplate):
    key = cuda.blockDim.z


class Cuda_gridDim_x(MacroTemplate):
    key = cuda.gridDim.x


class Cuda_gridDim_y(MacroTemplate):
    key = cuda.gridDim.y


class Cuda_gridDim_z(MacroTemplate):
    key = cuda.gridDim.z


class Cuda_shared_array(MacroTemplate):
    key = cuda.shared.array


class Cuda_local_array(MacroTemplate):
    key = cuda.local.array


class Cuda_const_arraylike(MacroTemplate):
    key = cuda.const.array_like


@intrinsic
class Cuda_syncthreads(ConcreteTemplate):
    key = cuda.syncthreads
    cases = [signature(types.none)]


@intrinsic
class Cuda_threadfence_device(ConcreteTemplate):
    key = cuda.threadfence
    cases = [signature(types.none)]

@intrinsic
class Cuda_threadfence_block(ConcreteTemplate):
    key = cuda.threadfence_block
    cases = [signature(types.none)]

@intrinsic
class Cuda_threadfence_system(ConcreteTemplate):
    key = cuda.threadfence_system
    cases = [signature(types.none)]


@intrinsic
class Cuda_atomic_add(AbstractTemplate):
    key = cuda.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic
class Cuda_atomic_max(AbstractTemplate):
    key = cuda.atomic.max

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        # Implementation presently supports float64 only,
        # so fail typing otherwise
        if ary.dtype != types.float64:
            return

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class Cuda_threadIdx(AttributeTemplate):
    key = types.Module(cuda.threadIdx)

    def resolve_x(self, mod):
        return types.Macro(Cuda_threadIdx_x)

    def resolve_y(self, mod):
        return types.Macro(Cuda_threadIdx_y)

    def resolve_z(self, mod):
        return types.Macro(Cuda_threadIdx_z)


@intrinsic_attr
class Cuda_blockIdx(AttributeTemplate):
    key = types.Module(cuda.blockIdx)

    def resolve_x(self, mod):
        return types.Macro(Cuda_blockIdx_x)

    def resolve_y(self, mod):
        return types.Macro(Cuda_blockIdx_y)

    def resolve_z(self, mod):
        return types.Macro(Cuda_blockIdx_z)


@intrinsic_attr
class Cuda_blockDim(AttributeTemplate):
    key = types.Module(cuda.blockDim)

    def resolve_x(self, mod):
        return types.Macro(Cuda_blockDim_x)

    def resolve_y(self, mod):
        return types.Macro(Cuda_blockDim_y)

    def resolve_z(self, mod):
        return types.Macro(Cuda_blockDim_z)


@intrinsic_attr
class Cuda_gridDim(AttributeTemplate):
    key = types.Module(cuda.gridDim)

    def resolve_x(self, mod):
        return types.Macro(Cuda_gridDim_x)

    def resolve_y(self, mod):
        return types.Macro(Cuda_gridDim_y)

    def resolve_z(self, mod):
        return types.Macro(Cuda_gridDim_z)


@intrinsic_attr
class CudaSharedModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.shared)

    def resolve_array(self, mod):
        return types.Macro(Cuda_shared_array)


@intrinsic_attr
class CudaConstModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.const)

    def resolve_array_like(self, mod):
        return types.Macro(Cuda_const_arraylike)


@intrinsic_attr
class CudaLocalModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.local)

    def resolve_array(self, mod):
        return types.Macro(Cuda_local_array)


@intrinsic_attr
class CudaAtomicTemplate(AttributeTemplate):
    key = types.Module(cuda.atomic)

    def resolve_add(self, mod):
        return types.Function(Cuda_atomic_add)

    def resolve_max(self, mod):
        return types.Function(Cuda_atomic_max)


@intrinsic_attr
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_grid(self, mod):
        return types.Macro(Cuda_grid)

    def resolve_gridsize(self, mod):
        return types.Macro(Cuda_gridsize)

    def resolve_threadIdx(self, mod):
        return types.Module(cuda.threadIdx)

    def resolve_blockIdx(self, mod):
        return types.Module(cuda.blockIdx)

    def resolve_blockDim(self, mod):
        return types.Module(cuda.blockDim)

    def resolve_gridDim(self, mod):
        return types.Module(cuda.gridDim)

    def resolve_shared(self, mod):
        return types.Module(cuda.shared)

    def resolve_syncthreads(self, mod):
        return types.Function(Cuda_syncthreads)

    def resolve_threadfence(self, mod):
        return types.Function(Cuda_threadfence_device)

    def resolve_threadfence_block(self, mod):
        return types.Function(Cuda_threadfence_block)

    def resolve_threadfence_system(self, mod):
        return types.Function(Cuda_threadfence_system)

    def resolve_atomic(self, mod):
        return types.Module(cuda.atomic)

    def resolve_const(self, mod):
        return types.Module(cuda.const)

    def resolve_local(self, mod):
        return types.Module(cuda.local)


intrinsic_global(cuda, types.Module(cuda))
## Forces the use of the cuda namespace by not recognizing individual the
## following as globals.
# intrinsic_global(cuda.grid, types.Function(Cuda_grid))
# intrinsic_global(cuda.gridsize, types.Function(Cuda_gridsize))
# intrinsic_global(cuda.threadIdx, types.Module(cuda.threadIdx))
# intrinsic_global(cuda.shared, types.Module(cuda.shared))
# intrinsic_global(cuda.shared.array, types.Function(Cuda_shared_array))
# intrinsic_global(cuda.syncthreads, types.Function(Cuda_syncthreads))
# intrinsic_global(cuda.atomic, types.Module(cuda.atomic))

