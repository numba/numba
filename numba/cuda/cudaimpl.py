from __future__ import print_function, absolute_import, division

from functools import reduce
import operator
import six

from llvmlite.llvmpy.core import Type
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba.targets.imputils import Registry
from numba import cgutils
from numba import types
from .cudadrv import nvvm
from . import nvvmutils, stubs

registry = Registry()
lower = registry.lower


@lower('ptx.grid.1d', types.intp)
def ptx_grid1d(context, builder, sig, args):
    assert len(args) == 1
    return nvvmutils.get_global_id(builder, dim=1)


@lower('ptx.grid.2d', types.intp)
def ptx_grid2d(context, builder, sig, args):
    assert len(args) == 1
    r1, r2 = nvvmutils.get_global_id(builder, dim=2)
    return cgutils.pack_array(builder, [r1, r2])


@lower('ptx.grid.3d', types.intp)
def ptx_grid3d(context, builder, sig, args):
    assert len(args) == 1
    r1, r2, r3 = nvvmutils.get_global_id(builder, dim=3)
    return cgutils.pack_array(builder, [r1, r2, r3])


@lower('ptx.gridsize.1d', types.intp)
def ptx_gridsize1d(context, builder, sig, args):
    assert len(args) == 1
    ntidx = nvvmutils.call_sreg(builder, "ntid.x")
    nctaidx = nvvmutils.call_sreg(builder, "nctaid.x")

    res = builder.mul(ntidx, nctaidx)
    return res


@lower('ptx.gridsize.2d', types.intp)
def ptx_gridsize2d(context, builder, sig, args):
    assert len(args) == 1
    ntidx = nvvmutils.call_sreg(builder, "ntid.x")
    nctaidx = nvvmutils.call_sreg(builder, "nctaid.x")

    ntidy = nvvmutils.call_sreg(builder, "ntid.y")
    nctaidy = nvvmutils.call_sreg(builder, "nctaid.y")

    r1 = builder.mul(ntidx, nctaidx)
    r2 = builder.mul(ntidy, nctaidy)
    return cgutils.pack_array(builder, [r1, r2])


@lower('ptx.gridsize.3d', types.intp)
def ptx_gridsize3d(context, builder, sig, args):
    assert len(args) == 1
    ntidx = nvvmutils.call_sreg(builder, "ntid.x")
    nctaidx = nvvmutils.call_sreg(builder, "nctaid.x")

    ntidy = nvvmutils.call_sreg(builder, "ntid.y")
    nctaidy = nvvmutils.call_sreg(builder, "nctaid.y")

    ntidz = nvvmutils.call_sreg(builder, "ntid.z")
    nctaidz = nvvmutils.call_sreg(builder, "nctaid.z")

    r1 = builder.mul(ntidx, nctaidx)
    r2 = builder.mul(ntidy, nctaidy)
    r3 = builder.mul(ntidz, nctaidz)
    return cgutils.pack_array(builder, [r1, r2, r3])


# -----------------------------------------------------------------------------

def ptx_sreg_template(sreg):
    def ptx_sreg_impl(context, builder, sig, args):
        assert not args
        return nvvmutils.call_sreg(builder, sreg)

    return ptx_sreg_impl


# Dynamic create all special register
for sreg in nvvmutils.SREG_MAPPING.keys():
    lower(sreg)(ptx_sreg_template(sreg))


# -----------------------------------------------------------------------------

@lower('ptx.cmem.arylike', types.Array)
def ptx_cmem_arylike(context, builder, sig, args):
    lmod = builder.module
    [arr] = args
    aryty = sig.return_type

    constvals = [
        context.get_constant(types.byte, i)
        for i in six.iterbytes(arr.tobytes(order='A'))
    ]
    constary = lc.Constant.array(Type.int(8), constvals)

    addrspace = nvvm.ADDRSPACE_CONSTANT
    gv = lmod.add_global_variable(constary.type, name="_cudapy_cmem",
                                  addrspace=addrspace)
    gv.linkage = lc.LINKAGE_INTERNAL
    gv.global_constant = True
    gv.initializer = constary

    # Preserve the underlying alignment
    lldtype = context.get_data_type(aryty.dtype)
    align = context.get_abi_sizeof(lldtype)
    gv.align = 2 ** (align - 1).bit_length()

    # Convert to generic address-space
    conv = nvvmutils.insert_addrspace_conv(lmod, Type.int(8), addrspace)
    addrspaceptr = gv.bitcast(Type.pointer(Type.int(8), addrspace))
    genptr = builder.call(conv, [addrspaceptr])

    # Create array object
    ary = context.make_array(aryty)(context, builder)
    kshape = [context.get_constant(types.intp, s) for s in arr.shape]
    kstrides = [context.get_constant(types.intp, s) for s in arr.strides]
    context.populate_array(ary,
                           data=builder.bitcast(genptr, ary.data.type),
                           shape=cgutils.pack_array(builder, kshape),
                           strides=cgutils.pack_array(builder, kstrides),
                           itemsize=ary.itemsize,
                           parent=ary.parent,
                           meminfo=None)

    return ary._getvalue()


_unique_smem_id = 0


def _get_unique_smem_id(name):
    """Due to bug with NVVM invalid internalizing of shared memory in the
    PTX output.  We can't mark shared memory to be internal. We have to
    ensure unique name is generated for shared memory symbol.
    """
    global _unique_smem_id
    _unique_smem_id += 1
    return "{0}_{1}".format(name, _unique_smem_id)


@lower('ptx.smem.alloc', types.intp, types.Any)
def ptx_smem_alloc_intp(context, builder, sig, args):
    length, dtype = args
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower('ptx.smem.alloc', types.UniTuple, types.Any)
def ptx_smem_alloc_array(context, builder, sig, args):
    shape, dtype = args
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@lower('ptx.lmem.alloc', types.intp, types.Any)
def ptx_lmem_alloc_intp(context, builder, sig, args):
    length, dtype = args
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name='_cudapy_lmem',
                          addrspace=nvvm.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@lower('ptx.lmem.alloc', types.UniTuple, types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    shape, dtype = args
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


@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i8, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f8, types.i4, types.i4)
def ptx_shfl_sync_i32(context, builder, sig, args):
    """
    The NVVM intrinsic for shfl only supports i32, but the cuda intrinsic function supports
    both 32 and 64 bit ints and floats, so for feature parity, i64, f32, and f64 are implemented.
    Floats by way of bitcasting the float to an int, then shuffling, then bitcasting back.
    And 64-bit values by packing them into 2 32bit values, shuffling thoose, and then packing back together.
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
        ptr = cgutils.get_item_pointer(builder, aryty, lary, indices)
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
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    elif dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float64(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@lower(stubs.atomic.max, types.Array, types.intp, types.Any)
@lower(stubs.atomic.max, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.max, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_max(context, builder, dtype, ptr, val):
    lmod = builder.module
    if dtype == types.float64:
        return builder.call(nvvmutils.declare_atomic_max_float64(lmod), (ptr, val))
    elif dtype == types.float32:
        return builder.call(nvvmutils.declare_atomic_max_float32(lmod), (ptr, val))
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
        return builder.call(nvvmutils.declare_atomic_min_float64(lmod), (ptr, val))
    elif dtype == types.float32:
        return builder.call(nvvmutils.declare_atomic_min_float32(lmod), (ptr, val))
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
    ptr = cgutils.get_item_pointer(builder, aryty, lary, (zero,))
    if aryty.dtype == types.int32:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_cas_int32(lmod), (ptr, old, val))
    else:
        raise TypeError('Unimplemented atomic compare_and_swap with %s array' % dtype)


# -----------------------------------------------------------------------------


def _get_target_data(context):
    return ll.create_target_data(nvvm.data_layout[context.address_size])


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace,
                   can_dynsized=False):
    elemcount = reduce(operator.mul, shape)
    lldtype = context.get_data_type(dtype)
    laryty = Type.array(lldtype, elemcount)

    if addrspace == nvvm.ADDRSPACE_LOCAL:
        # Special case local addrespace allocation to use alloca
        # NVVM is smart enough to only use local memory if no register is
        # available
        dataptr = cgutils.alloca_once(builder, laryty, name=symbol_name)
    else:
        lmod = builder.module

        # Create global variable in the requested address-space
        gvmem = lmod.add_global_variable(laryty, symbol_name, addrspace)
        # Specify alignment to avoid misalignment bug
        gvmem.align = context.get_abi_sizeof(lldtype)

        if elemcount <= 0:
            if can_dynsized:    # dynamic shared memory
                gvmem.linkage = lc.LINKAGE_EXTERNAL
            else:
                raise ValueError("array length <= 0")
        else:
            ## Comment out the following line to workaround a NVVM bug
            ## which generates a invalid symbol name when the linkage
            ## is internal and in some situation.
            ## See _get_unique_smem_id()
            # gvmem.linkage = lc.LINKAGE_INTERNAL

            gvmem.initializer = lc.Constant.undef(laryty)

        if dtype not in types.number_domain:
            raise TypeError("unsupported type: %s" % dtype)

        # Convert to generic address-space
        conv = nvvmutils.insert_addrspace_conv(lmod, Type.int(8), addrspace)
        addrspaceptr = gvmem.bitcast(Type.pointer(Type.int(8), addrspace))
        dataptr = builder.call(conv, [addrspaceptr])

    return _make_array(context, builder, dataptr, dtype, shape)


def _make_array(context, builder, dataptr, dtype, shape, layout='C'):
    ndim = len(shape)
    # Create array object
    aryty = types.Array(dtype=dtype, ndim=ndim, layout='C')
    ary = context.make_array(aryty)(context, builder)

    targetdata = _get_target_data(context)
    lldtype = context.get_data_type(dtype)
    itemsize = lldtype.get_abi_size(targetdata)
    # Compute strides
    rstrides = [itemsize]
    for i, lastsize in enumerate(reversed(shape[1:])):
        rstrides.append(lastsize * rstrides[-1])
    strides = [s for s in reversed(rstrides)]

    kshape = [context.get_constant(types.intp, s) for s in shape]
    kstrides = [context.get_constant(types.intp, s) for s in strides]

    context.populate_array(ary,
                           data=builder.bitcast(dataptr, ary.data.type),
                           shape=cgutils.pack_array(builder, kshape),
                           strides=cgutils.pack_array(builder, kstrides),
                           itemsize=context.get_constant(types.intp, itemsize),
                           meminfo=None)
    return ary._getvalue()
