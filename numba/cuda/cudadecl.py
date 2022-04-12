from numba.core import types, errors
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
                                       register_number_classes)
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         AbstractTemplate, CallableTemplate,
                                         signature, Registry)
from numba.cuda.types import dim3, grid_group
from numba import cuda


registry = Registry()
register = registry.register
register_attr = registry.register_attr
register_global = registry.register_global

register_number_classes(register_global)


class GridFunction(CallableTemplate):
    def generic(self):
        def typer(ndim):
            if not isinstance(ndim, types.IntegerLiteral):
                raise errors.RequireLiteralValue(ndim)
            val = ndim.literal_value
            if val == 1:
                restype = types.int32
            elif val in (2, 3):
                restype = types.UniTuple(types.int32, val)
            else:
                raise ValueError('argument can only be 1, 2, 3')
            return signature(restype, types.int32)
        return typer


@register
class Cuda_grid(GridFunction):
    key = cuda.grid


@register
class Cuda_gridsize(GridFunction):
    key = cuda.gridsize


class Cuda_array_decl(CallableTemplate):
    def generic(self):
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral)
                        for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        return typer


@register
class Cuda_shared_array(Cuda_array_decl):
    key = cuda.shared.array


@register
class Cuda_local_array(Cuda_array_decl):
    key = cuda.local.array


@register
class Cuda_const_array_like(CallableTemplate):
    key = cuda.const.array_like

    def generic(self):
        def typer(ndarray):
            return ndarray
        return typer


@register
class Cuda_syncthreads(ConcreteTemplate):
    key = cuda.syncthreads
    cases = [signature(types.none)]


@register
class Cuda_syncthreads_count(ConcreteTemplate):
    key = cuda.syncthreads_count
    cases = [signature(types.i4, types.i4)]


@register
class Cuda_syncthreads_and(ConcreteTemplate):
    key = cuda.syncthreads_and
    cases = [signature(types.i4, types.i4)]


@register
class Cuda_syncthreads_or(ConcreteTemplate):
    key = cuda.syncthreads_or
    cases = [signature(types.i4, types.i4)]


@register
class Cuda_threadfence_device(ConcreteTemplate):
    key = cuda.threadfence
    cases = [signature(types.none)]


@register
class Cuda_threadfence_block(ConcreteTemplate):
    key = cuda.threadfence_block
    cases = [signature(types.none)]


@register
class Cuda_threadfence_system(ConcreteTemplate):
    key = cuda.threadfence_system
    cases = [signature(types.none)]


@register
class Cuda_syncwarp(ConcreteTemplate):
    key = cuda.syncwarp
    cases = [signature(types.none), signature(types.none, types.i4)]


@register
class Cuda_cg_this_grid(ConcreteTemplate):
    key = cuda.cg.this_grid
    cases = [signature(grid_group)]


@register_attr
class CudaCgModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.cg)

    def resolve_this_grid(self, mod):
        return types.Function(Cuda_cg_this_grid)


class Cuda_grid_group_sync(AbstractTemplate):
    key = "GridGroup.sync"

    def generic(self, args, kws):
        return signature(types.int32, recvr=self.this)


@register_attr
class GridGroup_attrs(AttributeTemplate):
    key = grid_group

    def resolve_sync(self, mod):
        return types.BoundFunction(Cuda_grid_group_sync, grid_group)


@register
class Cuda_shfl_sync_intrinsic(ConcreteTemplate):
    key = cuda.shfl_sync_intrinsic
    cases = [
        signature(types.Tuple((types.i4, types.b1)),
                  types.i4, types.i4, types.i4, types.i4, types.i4),
        signature(types.Tuple((types.i8, types.b1)),
                  types.i4, types.i4, types.i8, types.i4, types.i4),
        signature(types.Tuple((types.f4, types.b1)),
                  types.i4, types.i4, types.f4, types.i4, types.i4),
        signature(types.Tuple((types.f8, types.b1)),
                  types.i4, types.i4, types.f8, types.i4, types.i4),
    ]


@register
class Cuda_vote_sync_intrinsic(ConcreteTemplate):
    key = cuda.vote_sync_intrinsic
    cases = [signature(types.Tuple((types.i4, types.b1)),
                       types.i4, types.i4, types.b1)]


@register
class Cuda_match_any_sync(ConcreteTemplate):
    key = cuda.match_any_sync
    cases = [
        signature(types.i4, types.i4, types.i4),
        signature(types.i4, types.i4, types.i8),
        signature(types.i4, types.i4, types.f4),
        signature(types.i4, types.i4, types.f8),
    ]


@register
class Cuda_match_all_sync(ConcreteTemplate):
    key = cuda.match_all_sync
    cases = [
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.i8),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f4),
        signature(types.Tuple((types.i4, types.b1)), types.i4, types.f8),
    ]


@register
class Cuda_activemask(ConcreteTemplate):
    key = cuda.activemask
    cases = [signature(types.uint32)]


@register
class Cuda_lanemask_lt(ConcreteTemplate):
    key = cuda.lanemask_lt
    cases = [signature(types.uint32)]


@register
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


@register
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


@register
class Cuda_hfma(ConcreteTemplate):
    key = cuda.fp16.hfma
    cases = [
        signature(types.float16, types.float16, types.float16, types.float16)
    ]


@register
class Cuda_cbrt(ConcreteTemplate):

    key = cuda.cbrt
    cases = [
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


@register
class Cuda_brev(ConcreteTemplate):
    key = cuda.brev
    cases = [
        signature(types.uint32, types.uint32),
        signature(types.uint64, types.uint64),
    ]


@register
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


@register
class Cuda_ffs(ConcreteTemplate):
    """
    Supported types from `llvm.cttz`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.ffs
    cases = [
        signature(types.uint32, types.int8),
        signature(types.uint32, types.int16),
        signature(types.uint32, types.int32),
        signature(types.uint32, types.int64),
        signature(types.uint32, types.uint8),
        signature(types.uint32, types.uint16),
        signature(types.uint32, types.uint32),
        signature(types.uint32, types.uint64),
    ]


@register
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


def _genfp16_unary(l_key):
    @register
    class Cuda_fp16_unary(ConcreteTemplate):
        key = l_key
        cases = [signature(types.float16, types.float16)]

    return Cuda_fp16_unary


def _genfp16_binary(l_key):
    @register
    class Cuda_fp16_binary(ConcreteTemplate):
        key = l_key
        cases = [signature(types.float16, types.float16, types.float16)]

    return Cuda_fp16_binary


Cuda_hadd = _genfp16_binary(cuda.fp16.hadd)
Cuda_hsub = _genfp16_binary(cuda.fp16.hsub)
Cuda_hmul = _genfp16_binary(cuda.fp16.hmul)
Cuda_hneg = _genfp16_unary(cuda.fp16.hneg)
Cuda_habs = _genfp16_unary(cuda.fp16.habs)


# generate atomic operations
def _gen(l_key, supported_types):
    @register
    class Cuda_atomic(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            ary, idx, val = args

            if ary.dtype not in supported_types:
                return

            if ary.ndim == 1:
                return signature(ary.dtype, ary, types.intp, ary.dtype)
            elif ary.ndim > 1:
                return signature(ary.dtype, ary, idx, ary.dtype)
    return Cuda_atomic


all_numba_types = (types.float64, types.float32,
                   types.int32, types.uint32,
                   types.int64, types.uint64)

integer_numba_types = (types.int32, types.uint32,
                       types.int64, types.uint64)

unsigned_int_numba_types = (types.uint32, types.uint64)

Cuda_atomic_add = _gen(cuda.atomic.add, all_numba_types)
Cuda_atomic_sub = _gen(cuda.atomic.sub, all_numba_types)
Cuda_atomic_max = _gen(cuda.atomic.max, all_numba_types)
Cuda_atomic_min = _gen(cuda.atomic.min, all_numba_types)
Cuda_atomic_nanmax = _gen(cuda.atomic.nanmax, all_numba_types)
Cuda_atomic_nanmin = _gen(cuda.atomic.nanmin, all_numba_types)
Cuda_atomic_and = _gen(cuda.atomic.and_, integer_numba_types)
Cuda_atomic_or = _gen(cuda.atomic.or_, integer_numba_types)
Cuda_atomic_xor = _gen(cuda.atomic.xor, integer_numba_types)
Cuda_atomic_inc = _gen(cuda.atomic.inc, unsigned_int_numba_types)
Cuda_atomic_dec = _gen(cuda.atomic.dec, unsigned_int_numba_types)
Cuda_atomic_exch = _gen(cuda.atomic.exch, integer_numba_types)


@register
class Cuda_atomic_compare_and_swap(AbstractTemplate):
    key = cuda.atomic.compare_and_swap

    def generic(self, args, kws):
        assert not kws
        ary, old, val = args
        dty = ary.dtype

        if dty in integer_numba_types and ary.ndim == 1:
            return signature(dty, ary, dty, dty)


@register
class Cuda_nanosleep(ConcreteTemplate):
    key = cuda.nanosleep

    cases = [signature(types.void, types.uint32)]


@register_attr
class Dim3_attrs(AttributeTemplate):
    key = dim3

    def resolve_x(self, mod):
        return types.int32

    def resolve_y(self, mod):
        return types.int32

    def resolve_z(self, mod):
        return types.int32


@register_attr
class CudaSharedModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.shared)

    def resolve_array(self, mod):
        return types.Function(Cuda_shared_array)


@register_attr
class CudaConstModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.const)

    def resolve_array_like(self, mod):
        return types.Function(Cuda_const_array_like)


@register_attr
class CudaLocalModuleTemplate(AttributeTemplate):
    key = types.Module(cuda.local)

    def resolve_array(self, mod):
        return types.Function(Cuda_local_array)


@register_attr
class CudaAtomicTemplate(AttributeTemplate):
    key = types.Module(cuda.atomic)

    def resolve_add(self, mod):
        return types.Function(Cuda_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Cuda_atomic_sub)

    def resolve_and_(self, mod):
        return types.Function(Cuda_atomic_and)

    def resolve_or_(self, mod):
        return types.Function(Cuda_atomic_or)

    def resolve_xor(self, mod):
        return types.Function(Cuda_atomic_xor)

    def resolve_inc(self, mod):
        return types.Function(Cuda_atomic_inc)

    def resolve_dec(self, mod):
        return types.Function(Cuda_atomic_dec)

    def resolve_exch(self, mod):
        return types.Function(Cuda_atomic_exch)

    def resolve_max(self, mod):
        return types.Function(Cuda_atomic_max)

    def resolve_min(self, mod):
        return types.Function(Cuda_atomic_min)

    def resolve_nanmin(self, mod):
        return types.Function(Cuda_atomic_nanmin)

    def resolve_nanmax(self, mod):
        return types.Function(Cuda_atomic_nanmax)

    def resolve_compare_and_swap(self, mod):
        return types.Function(Cuda_atomic_compare_and_swap)


@register_attr
class CudaFp16Template(AttributeTemplate):
    key = types.Module(cuda.fp16)

    def resolve_hadd(self, mod):
        return types.Function(Cuda_hadd)

    def resolve_hsub(self, mod):
        return types.Function(Cuda_hsub)

    def resolve_hmul(self, mod):
        return types.Function(Cuda_hmul)

    def resolve_hneg(self, mod):
        return types.Function(Cuda_hneg)

    def resolve_habs(self, mod):
        return types.Function(Cuda_habs)

    def resolve_hfma(self, mod):
        return types.Function(Cuda_hfma)


@register_attr
class CudaModuleTemplate(AttributeTemplate):
    key = types.Module(cuda)

    def resolve_grid(self, mod):
        return types.Function(Cuda_grid)

    def resolve_gridsize(self, mod):
        return types.Function(Cuda_gridsize)

    def resolve_cg(self, mod):
        return types.Module(cuda.cg)

    def resolve_threadIdx(self, mod):
        return dim3

    def resolve_blockIdx(self, mod):
        return dim3

    def resolve_blockDim(self, mod):
        return dim3

    def resolve_gridDim(self, mod):
        return dim3

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

    def resolve_cbrt(self, mod):
        return types.Function(Cuda_cbrt)

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

    def resolve_activemask(self, mod):
        return types.Function(Cuda_activemask)

    def resolve_lanemask_lt(self, mod):
        return types.Function(Cuda_lanemask_lt)

    def resolve_selp(self, mod):
        return types.Function(Cuda_selp)

    def resolve_nanosleep(self, mod):
        return types.Function(Cuda_nanosleep)

    def resolve_atomic(self, mod):
        return types.Module(cuda.atomic)

    def resolve_fp16(self, mod):
        return types.Module(cuda.fp16)

    def resolve_const(self, mod):
        return types.Module(cuda.const)

    def resolve_local(self, mod):
        return types.Module(cuda.local)


register_global(cuda, types.Module(cuda))
