from functools import reduce
import operator
import math

from llvmlite.llvmpy.core import Type, InlineAsm
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba.core.imputils import Registry
from numba.core.typing.npydecl import parse_dtype
from numba.core import types, cgutils
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs
from numba.cuda.types import dim3


registry = Registry()
lower = registry.lower
lower_attr = registry.lower_getattr


def initialize_dim3(builder, prefix):
    x = nvvmutils.call_sreg(builder, "%s.x" % prefix)
    y = nvvmutils.call_sreg(builder, "%s.y" % prefix)
    z = nvvmutils.call_sreg(builder, "%s.z" % prefix)
    return cgutils.pack_struct(builder, (x, y, z))


@lower_attr(types.Module(cuda), 'threadIdx')
def cuda_threadIdx(context, builder, sig, args):
    return initialize_dim3(builder, 'tid')


@lower_attr(types.Module(cuda), 'blockDim')
def cuda_blockDim(context, builder, sig, args):
    return initialize_dim3(builder, 'ntid')


@lower_attr(types.Module(cuda), 'blockIdx')
def cuda_blockIdx(context, builder, sig, args):
    return initialize_dim3(builder, 'ctaid')


@lower_attr(types.Module(cuda), 'gridDim')
def cuda_gridDim(context, builder, sig, args):
    return initialize_dim3(builder, 'nctaid')


@lower_attr(types.Module(cuda), 'laneid')
def cuda_laneid(context, builder, sig, args):
    return nvvmutils.call_sreg(builder, 'laneid')


@lower_attr(types.Module(cuda), 'warpsize')
def cuda_warpsize(context, builder, sig, args):
    return nvvmutils.call_sreg(builder, 'warpsize')


@lower_attr(dim3, 'x')
def dim3_x(context, builder, sig, args):
    return builder.extract_value(args, 0)


@lower_attr(dim3, 'y')
def dim3_y(context, builder, sig, args):
    return builder.extract_value(args, 1)


@lower_attr(dim3, 'z')
def dim3_z(context, builder, sig, args):
    return builder.extract_value(args, 2)


@lower(cuda.grid, types.int32)
def cuda_grid(context, builder, sig, args):
    restype = sig.return_type
    if restype == types.int32:
        return nvvmutils.get_global_id(builder, dim=1)
    elif isinstance(restype, types.UniTuple):
        ids = nvvmutils.get_global_id(builder, dim=restype.count)
        return cgutils.pack_array(builder, ids)
    else:
        raise ValueError('Unexpected return type %s from cuda.grid' % restype)


def _nthreads_for_dim(builder, dim):
    ntid = nvvmutils.call_sreg(builder, f"ntid.{dim}")
    nctaid = nvvmutils.call_sreg(builder, f"nctaid.{dim}")
    return builder.mul(ntid, nctaid)


@lower(cuda.gridsize, types.int32)
def cuda_gridsize(context, builder, sig, args):
    restype = sig.return_type
    nx = _nthreads_for_dim(builder, 'x')

    if restype == types.int32:
        return nx
    elif isinstance(restype, types.UniTuple):
        ny = _nthreads_for_dim(builder, 'y')

        if restype.count == 2:
            return cgutils.pack_array(builder, (nx, ny))
        elif restype.count == 3:
            nz = _nthreads_for_dim(builder, 'z')
            return cgutils.pack_array(builder, (nx, ny, nz))

    # Fallthrough to here indicates unexpected return type or tuple length
    raise ValueError('Unexpected return type %s of cuda.gridsize' % restype)


# -----------------------------------------------------------------------------

@lower(cuda.const.array_like, types.Array)
def cuda_const_array_like(context, builder, sig, args):
    # This is a no-op because CUDATargetContext.make_constant_array already
    # created the constant array.
    return args[0]


_unique_smem_id = 0


def _get_unique_smem_id(name):
    """Due to bug with NVVM invalid internalizing of shared memory in the
    PTX output.  We can't mark shared memory to be internal. We have to
    ensure unique name is generated for shared memory symbol.
    """
    global _unique_smem_id
    _unique_smem_id += 1
    return "{0}_{1}".format(name, _unique_smem_id)


@lower(cuda.shared.array, types.IntegerLiteral, types.Any)
def cuda_shared_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower(cuda.shared.array, types.Tuple, types.Any)
@lower(cuda.shared.array, types.UniTuple, types.Any)
def cuda_shared_array_tuple(context, builder, sig, args):
    shape = [ s.literal_value for s in sig.args[0] ]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower(cuda.local.array, types.IntegerLiteral, types.Any)
def cuda_local_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name='_cudapy_lmem',
                          addrspace=nvvm.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@lower(cuda.local.array, types.Tuple, types.Any)
@lower(cuda.local.array, types.UniTuple, types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    shape = [ s.literal_value for s in sig.args[0] ]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_cudapy_lmem',
                          addrspace=nvvm.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@lower(stubs.syncthreads)
def ptx_syncthreads(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.barrier0'
    lmod = builder.module
    fnty = Type.function(Type.void(), ())
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, ())
    return context.get_dummy_value()


@lower(stubs.syncthreads_count, types.i4)
def ptx_syncthreads_count(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.popc'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    sync = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(sync, args)


@lower(stubs.syncthreads_and, types.i4)
def ptx_syncthreads_and(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.and'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    sync = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(sync, args)


@lower(stubs.syncthreads_or, types.i4)
def ptx_syncthreads_or(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.or'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    sync = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(sync, args)


@lower(stubs.threadfence_block)
def ptx_threadfence_block(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.membar.cta'
    lmod = builder.module
    fnty = Type.function(Type.void(), ())
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, ())
    return context.get_dummy_value()


@lower(stubs.threadfence_system)
def ptx_threadfence_system(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.membar.sys'
    lmod = builder.module
    fnty = Type.function(Type.void(), ())
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, ())
    return context.get_dummy_value()


@lower(stubs.threadfence)
def ptx_threadfence_device(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.membar.gl'
    lmod = builder.module
    fnty = Type.function(Type.void(), ())
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, ())
    return context.get_dummy_value()


@lower(stubs.syncwarp, types.i4)
def ptx_warp_sync(context, builder, sig, args):
    fname = 'llvm.nvvm.bar.warp.sync'
    lmod = builder.module
    fnty = Type.function(Type.void(), (Type.int(32),))
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, args)
    return context.get_dummy_value()


@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i4, types.i4,
       types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i8, types.i4,
       types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f4, types.i4,
       types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f8, types.i4,
       types.i4)
def ptx_shfl_sync_i32(context, builder, sig, args):
    """
    The NVVM intrinsic for shfl only supports i32, but the cuda intrinsic
    function supports both 32 and 64 bit ints and floats, so for feature parity,
    i64, f32, and f64 are implemented. Floats by way of bitcasting the float to
    an int, then shuffling, then bitcasting back. And 64-bit values by packing
    them into 2 32bit values, shuffling thoose, and then packing back together.
    """
    mask, mode, value, index, clamp = args
    value_type = sig.args[2]
    if value_type in types.real_domain:
        value = builder.bitcast(value, Type.int(value_type.bitwidth))
    fname = 'llvm.nvvm.shfl.sync.i32'
    lmod = builder.module
    fnty = Type.function(
        Type.struct((Type.int(32), Type.int(1))),
        (Type.int(32), Type.int(32), Type.int(32), Type.int(32), Type.int(32))
    )
    func = lmod.get_or_insert_function(fnty, name=fname)
    if value_type.bitwidth == 32:
        ret = builder.call(func, (mask, mode, value, index, clamp))
        if value_type == types.float32:
            rv = builder.extract_value(ret, 0)
            pred = builder.extract_value(ret, 1)
            fv = builder.bitcast(rv, Type.float())
            ret = cgutils.make_anonymous_struct(builder, (fv, pred))
    else:
        value1 = builder.trunc(value, Type.int(32))
        value_lshr = builder.lshr(value, context.get_constant(types.i8, 32))
        value2 = builder.trunc(value_lshr, Type.int(32))
        ret1 = builder.call(func, (mask, mode, value1, index, clamp))
        ret2 = builder.call(func, (mask, mode, value2, index, clamp))
        rv1 = builder.extract_value(ret1, 0)
        rv2 = builder.extract_value(ret2, 0)
        pred = builder.extract_value(ret1, 1)
        rv1_64 = builder.zext(rv1, Type.int(64))
        rv2_64 = builder.zext(rv2, Type.int(64))
        rv_shl = builder.shl(rv2_64, context.get_constant(types.i8, 32))
        rv = builder.or_(rv_shl, rv1_64)
        if value_type == types.float64:
            rv = builder.bitcast(rv, Type.double())
        ret = cgutils.make_anonymous_struct(builder, (rv, pred))
    return ret


@lower(stubs.vote_sync_intrinsic, types.i4, types.i4, types.boolean)
def ptx_vote_sync(context, builder, sig, args):
    fname = 'llvm.nvvm.vote.sync'
    lmod = builder.module
    fnty = Type.function(Type.struct((Type.int(32), Type.int(1))),
                         (Type.int(32), Type.int(32), Type.int(1)))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)


@lower(stubs.match_any_sync, types.i4, types.i4)
@lower(stubs.match_any_sync, types.i4, types.i8)
@lower(stubs.match_any_sync, types.i4, types.f4)
@lower(stubs.match_any_sync, types.i4, types.f8)
def ptx_match_any_sync(context, builder, sig, args):
    mask, value = args
    width = sig.args[1].bitwidth
    if sig.args[1] in types.real_domain:
        value = builder.bitcast(value, Type.int(width))
    fname = 'llvm.nvvm.match.any.sync.i{}'.format(width)
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32), Type.int(width)))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, (mask, value))


@lower(stubs.match_all_sync, types.i4, types.i4)
@lower(stubs.match_all_sync, types.i4, types.i8)
@lower(stubs.match_all_sync, types.i4, types.f4)
@lower(stubs.match_all_sync, types.i4, types.f8)
def ptx_match_all_sync(context, builder, sig, args):
    mask, value = args
    width = sig.args[1].bitwidth
    if sig.args[1] in types.real_domain:
        value = builder.bitcast(value, Type.int(width))
    fname = 'llvm.nvvm.match.all.sync.i{}'.format(width)
    lmod = builder.module
    fnty = Type.function(Type.struct((Type.int(32), Type.int(1))),
                         (Type.int(32), Type.int(width)))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, (mask, value))


@lower(stubs.popc, types.Any)
def ptx_popc(context, builder, sig, args):
    return builder.ctpop(args[0])


@lower(stubs.fma, types.Any, types.Any, types.Any)
def ptx_fma(context, builder, sig, args):
    return builder.fma(*args)


@lower(stubs.brev, types.u4)
def ptx_brev_u4(context, builder, sig, args):
    # FIXME the llvm.bitreverse.i32 intrinsic isn't supported by nvcc
    # return builder.bitreverse(args[0])

    fn = builder.module.get_or_insert_function(
        lc.Type.function(lc.Type.int(32), (lc.Type.int(32),)),
        '__nv_brev')
    return builder.call(fn, args)


@lower(stubs.brev, types.u8)
def ptx_brev_u8(context, builder, sig, args):
    # FIXME the llvm.bitreverse.i64 intrinsic isn't supported by nvcc
    # return builder.bitreverse(args[0])

    fn = builder.module.get_or_insert_function(
        lc.Type.function(lc.Type.int(64), (lc.Type.int(64),)),
        '__nv_brevll')
    return builder.call(fn, args)


@lower(stubs.clz, types.Any)
def ptx_clz(context, builder, sig, args):
    return builder.ctlz(
        args[0],
        context.get_constant(types.boolean, 0))


@lower(stubs.ffs, types.Any)
def ptx_ffs(context, builder, sig, args):
    return builder.cttz(
        args[0],
        context.get_constant(types.boolean, 0))


@lower(stubs.selp, types.Any, types.Any, types.Any)
def ptx_selp(context, builder, sig, args):
    test, a, b = args
    return builder.select(test, a, b)


@lower(max, types.f4, types.f4)
def ptx_max_f4(context, builder, sig, args):
    fn = builder.module.get_or_insert_function(
        lc.Type.function(
            lc.Type.float(),
            (lc.Type.float(), lc.Type.float())),
        '__nv_fmaxf')
    return builder.call(fn, args)


@lower(max, types.f8, types.f4)
@lower(max, types.f4, types.f8)
@lower(max, types.f8, types.f8)
def ptx_max_f8(context, builder, sig, args):
    fn = builder.module.get_or_insert_function(
        lc.Type.function(
            lc.Type.double(),
            (lc.Type.double(), lc.Type.double())),
        '__nv_fmax')

    return builder.call(fn, [
        context.cast(builder, args[0], sig.args[0], types.double),
        context.cast(builder, args[1], sig.args[1], types.double),
    ])


@lower(min, types.f4, types.f4)
def ptx_min_f4(context, builder, sig, args):
    fn = builder.module.get_or_insert_function(
        lc.Type.function(
            lc.Type.float(),
            (lc.Type.float(), lc.Type.float())),
        '__nv_fminf')
    return builder.call(fn, args)


@lower(min, types.f8, types.f4)
@lower(min, types.f4, types.f8)
@lower(min, types.f8, types.f8)
def ptx_min_f8(context, builder, sig, args):
    fn = builder.module.get_or_insert_function(
        lc.Type.function(
            lc.Type.double(),
            (lc.Type.double(), lc.Type.double())),
        '__nv_fmin')

    return builder.call(fn, [
        context.cast(builder, args[0], sig.args[0], types.double),
        context.cast(builder, args[1], sig.args[1], types.double),
    ])


@lower(round, types.f4)
@lower(round, types.f8)
def ptx_round(context, builder, sig, args):
    fn = builder.module.get_or_insert_function(
        lc.Type.function(
            lc.Type.int(64),
            (lc.Type.double(),)),
        '__nv_llrint')
    return builder.call(fn, [
        context.cast(builder, args[0], sig.args[0], types.double),
    ])


def gen_deg_rad(const):
    def impl(context, builder, sig, args):
        argty, = sig.args
        factor = context.get_constant(argty, const)
        return builder.fmul(factor, args[0])
    return impl


_deg2rad = math.pi / 180.
_rad2deg = 180. / math.pi
lower(math.radians, types.f4)(gen_deg_rad(_deg2rad))
lower(math.radians, types.f8)(gen_deg_rad(_deg2rad))
lower(math.degrees, types.f4)(gen_deg_rad(_rad2deg))
lower(math.degrees, types.f8)(gen_deg_rad(_rad2deg))


def _normalize_indices(context, builder, indty, inds):
    """
    Convert integer indices into tuple of intp
    """
    if indty in types.integer_domain:
        indty = types.UniTuple(dtype=indty, count=1)
        indices = [inds]
    else:
        indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
    indices = [context.cast(builder, i, t, types.intp)
               for t, i in zip(indty, indices)]
    return indty, indices


def _atomic_dispatcher(dispatch_fn):
    def imp(context, builder, sig, args):
        # The common argument handling code
        aryty, indty, valty = sig.args
        ary, inds, val = args
        dtype = aryty.dtype

        indty, indices = _normalize_indices(context, builder, indty, inds)

        if dtype != valty:
            raise TypeError("expect %s but got %s" % (dtype, valty))

        if aryty.ndim != len(indty):
            raise TypeError("indexing %d-D array with %d-D index" %
                            (aryty.ndim, len(indty)))

        lary = context.make_array(aryty)(context, builder, ary)
        ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices)
        # dispatcher to implementation base on dtype
        return dispatch_fn(context, builder, dtype, ptr, val)
    return imp


@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
@lower(stubs.atomic.add, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_add_tuple(context, builder, dtype, ptr, val):
    if dtype == types.float32:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod),
                            (ptr, val))
    elif dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float64(lmod),
                            (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@lower(stubs.atomic.max, types.Array, types.intp, types.Any)
@lower(stubs.atomic.max, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.max, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_max(context, builder, dtype, ptr, val):
    lmod = builder.module
    if dtype == types.float64:
        return builder.call(nvvmutils.declare_atomic_max_float64(lmod),
                            (ptr, val))
    elif dtype == types.float32:
        return builder.call(nvvmutils.declare_atomic_max_float32(lmod),
                            (ptr, val))
    elif dtype in (types.int32, types.int64):
        return builder.atomic_rmw('max', ptr, val, ordering='monotonic')
    elif dtype in (types.uint32, types.uint64):
        return builder.atomic_rmw('umax', ptr, val, ordering='monotonic')
    else:
        raise TypeError('Unimplemented atomic max with %s array' % dtype)


@lower(stubs.atomic.min, types.Array, types.intp, types.Any)
@lower(stubs.atomic.min, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.min, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_min(context, builder, dtype, ptr, val):
    lmod = builder.module
    if dtype == types.float64:
        return builder.call(nvvmutils.declare_atomic_min_float64(lmod),
                            (ptr, val))
    elif dtype == types.float32:
        return builder.call(nvvmutils.declare_atomic_min_float32(lmod),
                            (ptr, val))
    elif dtype in (types.int32, types.int64):
        return builder.atomic_rmw('min', ptr, val, ordering='monotonic')
    elif dtype in (types.uint32, types.uint64):
        return builder.atomic_rmw('umin', ptr, val, ordering='monotonic')
    else:
        raise TypeError('Unimplemented atomic min with %s array' % dtype)


@lower(stubs.atomic.compare_and_swap, types.Array, types.Any, types.Any)
def ptx_atomic_cas_tuple(context, builder, sig, args):
    aryty, oldty, valty = sig.args
    ary, old, val = args
    dtype = aryty.dtype

    lary = context.make_array(aryty)(context, builder, ary)
    zero = context.get_constant(types.intp, 0)
    ptr = cgutils.get_item_pointer(context, builder, aryty, lary, (zero,))
    if aryty.dtype == types.int32:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_cas_int32(lmod),
                            (ptr, old, val))
    else:
        raise TypeError('Unimplemented atomic compare_and_swap '
                        'with %s array' % dtype)


# -----------------------------------------------------------------------------


def _get_target_data(context):
    return ll.create_target_data(nvvm.data_layout[context.address_size])


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace,
                   can_dynsized=False):
    elemcount = reduce(operator.mul, shape)

    # Check for valid shape for this type of allocation
    dynamic_smem = elemcount <= 0 and can_dynsized
    if elemcount <= 0 and not dynamic_smem:
        raise ValueError("array length <= 0")

    # Check that we support the requested dtype
    other_supported_type = isinstance(dtype, (types.Record, types.Boolean))
    if dtype not in types.number_domain and not other_supported_type:
        raise TypeError("unsupported type: %s" % dtype)

    lldtype = context.get_data_type(dtype)
    laryty = Type.array(lldtype, elemcount)

    if addrspace == nvvm.ADDRSPACE_LOCAL:
        # Special case local address space allocation to use alloca
        # NVVM is smart enough to only use local memory if no register is
        # available
        dataptr = cgutils.alloca_once(builder, laryty, name=symbol_name)
    else:
        lmod = builder.module

        # Create global variable in the requested address space
        gvmem = lmod.add_global_variable(laryty, symbol_name, addrspace)
        # Specify alignment to avoid misalignment bug
        align = context.get_abi_sizeof(lldtype)
        # Alignment is required to be a power of 2 for shared memory. If it is
        # not a power of 2 (e.g. for a Record array) then round up accordingly.
        gvmem.align = 1 << (align - 1 ).bit_length()

        if dynamic_smem:
            gvmem.linkage = lc.LINKAGE_EXTERNAL
        else:
            ## Comment out the following line to workaround a NVVM bug
            ## which generates a invalid symbol name when the linkage
            ## is internal and in some situation.
            ## See _get_unique_smem_id()
            # gvmem.linkage = lc.LINKAGE_INTERNAL

            gvmem.initializer = lc.Constant.undef(laryty)

        # Convert to generic address-space
        conv = nvvmutils.insert_addrspace_conv(lmod, Type.int(8), addrspace)
        addrspaceptr = gvmem.bitcast(Type.pointer(Type.int(8), addrspace))
        dataptr = builder.call(conv, [addrspaceptr])

    targetdata = _get_target_data(context)
    lldtype = context.get_data_type(dtype)
    itemsize = lldtype.get_abi_size(targetdata)

    # Compute strides
    rstrides = [itemsize]
    for i, lastsize in enumerate(reversed(shape[1:])):
        rstrides.append(lastsize * rstrides[-1])
    strides = [s for s in reversed(rstrides)]
    kstrides = [context.get_constant(types.intp, s) for s in strides]

    # Compute shape
    if dynamic_smem:
        # Compute the shape based on the dynamic shared memory configuration.
        # Unfortunately NVVM does not provide an intrinsic for the
        # %dynamic_smem_size register, so we must read it using inline
        # assembly.
        get_dynshared_size = InlineAsm.get(Type.function(Type.int(), []),
                                           "mov.u32 $0, %dynamic_smem_size;",
                                           '=r', side_effect=True)
        dynsmem_size = builder.zext(builder.call(get_dynshared_size, []),
                                    Type.int(width=64))
        # Only 1-D dynamic shared memory is supported so the following is a
        # sufficient construction of the shape
        kitemsize = context.get_constant(types.intp, itemsize)
        kshape = [builder.udiv(dynsmem_size, kitemsize)]
    else:
        kshape = [context.get_constant(types.intp, s) for s in shape]

    # Create array object
    ndim = len(shape)
    aryty = types.Array(dtype=dtype, ndim=ndim, layout='C')
    ary = context.make_array(aryty)(context, builder)

    context.populate_array(ary,
                           data=builder.bitcast(dataptr, ary.data.type),
                           shape=cgutils.pack_array(builder, kshape),
                           strides=cgutils.pack_array(builder, kstrides),
                           itemsize=context.get_constant(types.intp, itemsize),
                           meminfo=None)
    return ary._getvalue()
