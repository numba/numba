from numba.core import types
from numba.core.typing.npydecl import register_number_classes
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         AbstractTemplate, MacroTemplate,
                                         CallableTemplate,
                                         signature, Registry)
from numba.core.extending import (typeof_impl, register_model, models,
                                  overload_attribute)
from numba import cuda


registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global

register_number_classes(intrinsic_global)


class GridFunction(CallableTemplate):
    def generic(self):
        def typer(ndim):
            val = ndim.literal_value
            if val == 1:
                restype = types.int32
            elif val in (2, 3):
                restype = types.UniTuple(types.int32, val)
            else:
                raise ValueError('argument can only be 1, 2, 3')
            return signature(restype, types.int32)
        return typer


@intrinsic
class Cuda_grid(GridFunction):
    key = cuda.grid


@intrinsic
class Cuda_gridsize(GridFunction):
    key = cuda.gridsize


def _parse_shape(shape):
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        if all(isinstance(s, types.Integer) for s in shape):
            ndim = len(shape)
    return ndim

def _parse_dtype(dtype):
    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype
    elif isinstance(dtype, types.TypeRef):
        return dtype.instance_type


@intrinsic
class Cuda_shared_array(CallableTemplate):
    key = cuda.shared.array

    def generic(self):
        def typer(shape, dtype):
            ndim = _parse_shape(shape)
            nb_dtype = _parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        return typer


class Cuda_local_array(MacroTemplate):
    key = cuda.local.array


class Cuda_const_arraylike(MacroTemplate):
    key = cuda.const.array_like


@intrinsic
class Cuda_syncthreads(ConcreteTemplate):
    key = cuda.syncthreads
    cases = [signature(types.none)]


@intrinsic
class Cuda_syncthreads_count(ConcreteTemplate):
    key = cuda.syncthreads_count
    cases = [signature(types.i4, types.i4)]


@intrinsic
class Cuda_syncthreads_and(ConcreteTemplate):
    key = cuda.syncthreads_and
    cases = [signature(types.i4, types.i4)]


@intrinsic
class Cuda_syncthreads_or(ConcreteTemplate):
    key = cuda.syncthreads_or
    cases = [signature(types.i4, types.i4)]


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
class Cuda_syncwarp(ConcreteTemplate):
    key = cuda.syncwarp
    cases = [signature(types.none, types.i4)]


@intrinsic
class Cuda_shfl_sync_intrinsic(ConcreteTemplate):
    key = cuda.shfl_sync_intrinsic
    cases = [
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4, types.i4, types.i4, types.i4),
        signature(types.Tuple((types.i8, types.b1)), types.i4, types.i4, types.i8, types.i4, types.i4),
        signature(types.Tuple((types.f4, types.b1)), types.i4, types.i4, types.f4, types.i4, types.i4),
        signature(types.Tuple((types.f8, types.b1)), types.i4, types.i4, types.f8, types.i4, types.i4),
    ]


@intrinsic
class Cuda_vote_sync_intrinsic(ConcreteTemplate):
    key = cuda.vote_sync_intrinsic
    cases = [signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4, types.b1)]


@intrinsic
class Cuda_match_any_sync(ConcreteTemplate):
    key = cuda.match_any_sync
    cases = [
        signature(types.i4, types.i4, types.i4),
        signature(types.i4, types.i4, types.i8),
        signature(types.i4, types.i4, types.f4),
        signature(types.i4, types.i4, types.f8),
    ]


@intrinsic
class Cuda_match_all_sync(ConcreteTemplate):
    key = cuda.match_all_sync
    cases = [
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i8),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f8),
    ]


@intrinsic
class Cuda_popc(ConcreteTemplate):
    """
    Supported types from `llvm.popc`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.popc
    cases = [
        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]

@intrinsic
class Cuda_fma(ConcreteTemplate):
    """
    Supported types from `llvm.fma`
    [here](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#standard-c-library-intrinics)
    """
    key = cuda.fma
    cases = [
        signature(types.float32, types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64, types.float64),
    ]


@intrinsic
class Cuda_brev(ConcreteTemplate):
    key = cuda.brev
    cases = [
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@intrinsic
class Cuda_clz(ConcreteTemplate):
    """
    Supported types from `llvm.ctlz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.clz
    cases = [
        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@intrinsic
class Cuda_ffs(ConcreteTemplate):
    """
    Supported types from `llvm.cttz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.ffs
    cases = [
        signature(types.int8, types.int8),
        signature(types.int16, types.int16),
        signature(types.int32, types.int32),
        signature(types.int64, types.int64),
        signature(types.uint8, types.uint8),
        signature(types.uint16, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@intrinsic
class Cuda_selp(AbstractTemplate):
    key = cuda.selp

    def generic(self, args, kws):
        assert not kws
        test, a, b = args


        # per docs
        # http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp
        supported_types = (types.float64, types.float32,
                           types.int16, types.uint16,
                           types.int32, types.uint32,
                           types.int64, types.uint64)

        if a != b or a not in supported_types:
            return

        return signature(a, test, a, a)


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


class Cuda_atomic_maxmin(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        # Implementation presently supports:
        # float64, float32, int32, int64, uint32, uint64 only,
        # so fail typing otherwise
        supported_types = (types.float64, types.float32,
                           types.int32, types.uint32,
                           types.int64, types.uint64)
        if ary.dtype not in supported_types:
            return

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic
class Cuda_atomic_max(Cuda_atomic_maxmin):
    key = cuda.atomic.max


@intrinsic
class Cuda_atomic_min(Cuda_atomic_maxmin):
    key = cuda.atomic.min


@intrinsic
class Cuda_atomic_compare_and_swap(AbstractTemplate):
    key = cuda.atomic.compare_and_swap

    def generic(self, args, kws):
        assert not kws
        ary, old, val = args
        dty = ary.dtype
        # only support int32
        if dty == types.int32 and ary.ndim == 1:
            return signature(dty, ary, dty, dty)


class Dim3Type(types.Type):
    def __init__(self):
        super().__init__(name='Dim3')

dim3_type = Dim3Type()

@typeof_impl.register(cuda.Dim3)
def typeof_dim3(val, c):
    return dim3_type


@register_model(Dim3Type)
class Dim3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.int32),
            ('y', types.int32),
            ('z', types.int32)
        ]
        super().__init__(dmm, fe_type, members)

@intrinsic_attr
class Dim3_attrs(AttributeTemplate):
    key = dim3_type

    def resolve_x(self, mod):
        return types.int32

    def resolve_y(self, mod):
        return types.int32

    def resolve_z(self, mod):
        return types.int32


@intrinsic_attr
class CudaSharedModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.shared)

    def resolve_array(self, mod):
        return types.Function(Cuda_shared_array)


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

    def resolve_min(self, mod):
        return types.Function(Cuda_atomic_min)

    def resolve_compare_and_swap(self, mod):
        return types.Function(Cuda_atomic_compare_and_swap)


@intrinsic_attr
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_grid(self, mod):
        return types.Function(Cuda_grid)

    def resolve_gridsize(self, mod):
        return types.Function(Cuda_gridsize)

    def resolve_threadIdx(self, mod):
        return dim3_type

    def resolve_blockIdx(self, mod):
        return dim3_type

    def resolve_blockDim(self, mod):
        return dim3_type

    def resolve_gridDim(self, mod):
        return dim3_type

    def resolve_warpsize(self, mod):
        return types.int32

    def resolve_laneid(self, mod):
        return types.int32

    def resolve_shared(self, mod):
        return types.Module(cuda.shared)

    def resolve_popc(self, mod):
        return types.Function(Cuda_popc)

    def resolve_brev(self, mod):
        return types.Function(Cuda_brev)

    def resolve_clz(self, mod):
        return types.Function(Cuda_clz)

    def resolve_ffs(self, mod):
        return types.Function(Cuda_ffs)

    def resolve_fma(self, mod):
        return types.Function(Cuda_fma)

    def resolve_syncthreads(self, mod):
        return types.Function(Cuda_syncthreads)

    def resolve_syncthreads_count(self, mod):
        return types.Function(Cuda_syncthreads_count)

    def resolve_syncthreads_and(self, mod):
        return types.Function(Cuda_syncthreads_and)

    def resolve_syncthreads_or(self, mod):
        return types.Function(Cuda_syncthreads_or)

    def resolve_threadfence(self, mod):
        return types.Function(Cuda_threadfence_device)

    def resolve_threadfence_block(self, mod):
        return types.Function(Cuda_threadfence_block)

    def resolve_threadfence_system(self, mod):
        return types.Function(Cuda_threadfence_system)

    def resolve_syncwarp(self, mod):
        return types.Function(Cuda_syncwarp)

    def resolve_shfl_sync_intrinsic(self, mod):
        return types.Function(Cuda_shfl_sync_intrinsic)

    def resolve_vote_sync_intrinsic(self, mod):
        return types.Function(Cuda_vote_sync_intrinsic)

    def resolve_match_any_sync(self, mod):
        return types.Function(Cuda_match_any_sync)

    def resolve_match_all_sync(self, mod):
        return types.Function(Cuda_match_all_sync)

    def resolve_selp(self, mod):
        return types.Function(Cuda_selp)

    def resolve_atomic(self, mod):
        return types.Module(cuda.atomic)

    def resolve_const(self, mod):
        return types.Module(cuda.const)

    def resolve_local(self, mod):
        return types.Module(cuda.local)


intrinsic_global(cuda, types.Module(cuda))
