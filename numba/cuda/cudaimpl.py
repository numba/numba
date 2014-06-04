from __future__ import print_function, absolute_import, division
from functools import reduce
import operator
from llvm.core import Type
import llvm.core as lc
import llvm.ee as le
from numba.targets.imputils import implement, Registry
from numba import cgutils
from numba import types
from .cudadrv import nvvm
from . import nvvmutils, stubs

registry = Registry()
register = registry.register


@register
@implement('ptx.grid.1d', types.intp)
def ptx_grid1d(context, builder, sig, args):
    assert len(args) == 1
    return nvvmutils.get_global_id(builder, dim=1)


@register
@implement('ptx.grid.2d', types.intp)
def ptx_grid2d(context, builder, sig, args):
    assert len(args) == 1
    r1, r2 = nvvmutils.get_global_id(builder, dim=2)
    return cgutils.pack_array(builder, [r1, r2])


@register
@implement('ptx.gridsize.1d', types.intp)
def ptx_gridsize1d(context, builder, sig, args):
    assert len(args) == 1
    ntidx = nvvmutils.call_sreg(builder, "ntid.x")
    nctaidx = nvvmutils.call_sreg(builder, "nctaid.x")

    res = builder.mul(ntidx, nctaidx)
    return res


@register
@implement('ptx.gridsize.2d', types.intp)
def ptx_gridsize2d(context, builder, sig, args):
    assert len(args) == 1
    ntidx = nvvmutils.call_sreg(builder, "ntid.x")
    nctaidx = nvvmutils.call_sreg(builder, "nctaid.x")

    ntidy = nvvmutils.call_sreg(builder, "ntid.y")
    nctaidy = nvvmutils.call_sreg(builder, "nctaid.y")

    r1 = builder.mul(ntidx, nctaidx)
    r2 = builder.mul(ntidy, nctaidy)
    return cgutils.pack_array(builder, [r1, r2])


# -----------------------------------------------------------------------------

def ptx_sreg_template(sreg):
    def ptx_sreg_impl(context, builder, sig, args):
        assert not args
        return nvvmutils.call_sreg(builder, sreg)

    return ptx_sreg_impl


# Dynamic create all special register
for sreg in nvvmutils.SREG_MAPPING.keys():
    register(implement(sreg)(ptx_sreg_template(sreg)))


# -----------------------------------------------------------------------------

@register
@implement('ptx.cmem.arylike', types.Kind(types.Array))
def ptx_cmem_arylike(context, builder, sig, args):
    lmod = cgutils.get_module(builder)
    [arr] = args
    flat = arr.flatten(order='A')
    aryty = sig.return_type
    dtype = aryty.dtype

    if isinstance(dtype, types.Complex):
        elemtype = (types.float32
                    if dtype == types.complex64
                    else types.float64)
        constvals = []
        for i in range(flat.size):
            elem = flat[i]
            real = context.get_constant(elemtype, elem.real)
            imag = context.get_constant(elemtype, elem.imag)
            constvals.extend([real, imag])

    elif dtype in types.number_domain:
        constvals = [context.get_constant(dtype, flat[i])
                     for i in range(flat.size)]

    else:
        raise TypeError("unsupport type: %s" % dtype)

    constary = lc.Constant.array(constvals[0].type, constvals)

    addrspace = nvvm.ADDRSPACE_CONSTANT
    gv = lmod.add_global_variable(constary.type, name="_cudapy_cmem",
                                  addrspace=addrspace)
    gv.linkage = lc.LINKAGE_INTERNAL
    gv.global_constant = True
    gv.initializer = constary

    # Convert to generic address-space
    conv = nvvmutils.insert_addrspace_conv(lmod, Type.int(8), addrspace)
    addrspaceptr = gv.bitcast(Type.pointer(Type.int(8), addrspace))
    genptr = builder.call(conv, [addrspaceptr])

    # Create array object
    ary = context.make_array(aryty)(context, builder)
    ary.data = builder.bitcast(genptr, ary.data.type)

    kshape = [context.get_constant(types.intp, s) for s in arr.shape]
    kstrides = [context.get_constant(types.intp, s) for s in arr.strides]

    ary.shape = cgutils.pack_array(builder, kshape)
    ary.strides = cgutils.pack_array(builder, kstrides)

    return ary._getvalue()


@register
@implement('ptx.smem.alloc', types.intp, types.Any)
def ptx_smem_alloc_intp(context, builder, sig, args):
    length, dtype = args
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name='_cudapy_smem',
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@register
@implement('ptx.smem.alloc', types.Kind(types.UniTuple), types.Any)
def ptx_smem_alloc_array(context, builder, sig, args):
    shape, dtype = args
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_cudapy_smem',
                          addrspace=nvvm.ADDRSPACE_SHARED,
                          can_dynsized=True)


@register
@implement('ptx.lmem.alloc', types.intp, types.Any)
def ptx_lmem_alloc_intp(context, builder, sig, args):
    length, dtype = args
    return _generic_array(context, builder, shape=(length,), dtype=dtype,
                          symbol_name='_cudapy_lmem',
                          addrspace=nvvm.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@register
@implement('ptx.lmem.alloc', types.Kind(types.UniTuple), types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    shape, dtype = args
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_cudapy_lmem',
                          addrspace=nvvm.ADDRSPACE_LOCAL,
                          can_dynsized=False)


@register
@implement(stubs.syncthreads)
def ptx_syncthreads(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.barrier0'
    lmod = cgutils.get_module(builder)
    fnty = Type.function(Type.void(), ())
    sync = lmod.get_or_insert_function(fnty, name=fname)
    builder.call(sync, ())
    return context.get_dummy_value()


@register
@implement(stubs.atomic.add, types.Kind(types.Array), types.intp, types.Any)
def ptx_atomic_add_intp(context, builder, sig, args):
    aryty, indty, valty = sig.args
    ary, ind, val = args
    dtype = aryty.dtype

    if dtype != valty:
        raise TypeError("expect %s but got %s" % (dtype, valty))
    if aryty.ndim != 1:
        raise TypeError("indexing %d-D array with 1-D index" % (aryty.ndim,))

    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(builder, aryty, lary, [ind])

    if aryty.dtype == types.float32:
        lmod = cgutils.get_module(builder)
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@register
@implement(stubs.atomic.add, types.Kind(types.Array),
           types.Kind(types.UniTuple), types.Any)
def ptx_atomic_add_unituple(context, builder, sig, args):
    aryty, indty, valty = sig.args
    ary, inds, val = args
    dtype = aryty.dtype

    indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
    indices = [context.cast(builder, i, t, types.intp)
               for t, i in zip(indty, indices)]

    if dtype != valty:
        raise TypeError("expect %s but got %s" % (dtype, valty))

    if aryty.ndim != len(indty):
        raise TypeError("indexing %d-D array with %d-D index" %
                        (aryty.ndim, len(indty)))

    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(builder, aryty, lary, indices)

    if aryty.dtype == types.float32:
        lmod = cgutils.get_module(builder)
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@register
@implement(stubs.atomic.add, types.Kind(types.Array),
           types.Kind(types.Tuple), types.Any)
def ptx_atomic_add_tuple(context, builder, sig, args):
    aryty, indty, valty = sig.args
    ary, inds, val = args
    dtype = aryty.dtype

    indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
    indices = [context.cast(builder, i, t, types.intp)
               for t, i in zip(indty, indices)]

    if dtype != valty:
        raise TypeError("expect %s but got %s" % (dtype, valty))

    if aryty.ndim != len(indty):
        raise TypeError("indexing %d-D array with %d-D index" %
                        (aryty.ndim, len(indty)))

    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(builder, aryty, lary, indices)

    if aryty.dtype == types.float32:
        lmod = cgutils.get_module(builder)
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


# -----------------------------------------------------------------------------


def _get_target_data(context):
    return le.TargetData.new(nvvm.data_layout[context.address_size])


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace,
                   can_dynsized=False):
    elemcount = reduce(operator.mul, shape)
    lldtype = context.get_data_type(dtype)
    laryty = Type.array(lldtype, elemcount)

    if addrspace == nvvm.ADDRSPACE_LOCAL:
        # Special case local addrespace allocation to use alloca
        # NVVM is smart enough to only use local memory if no register is
        # available
        dataptr = builder.alloca(laryty, name=symbol_name)
    else:
        lmod = cgutils.get_module(builder)

        # Create global variable in the requested address-space
        gvmem = lmod.add_global_variable(laryty, symbol_name, addrspace)

        if elemcount <= 0:
            if can_dynsized:    # dynamic shared memory
                gvmem.linkage = lc.LINKAGE_EXTERNAL
            else:
                raise ValueError("array length <= 0")
        else:
            gvmem.linkage = lc.LINKAGE_INTERNAL
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
    ary.data = builder.bitcast(dataptr, ary.data.type)

    targetdata = _get_target_data(context)
    lldtype = context.get_data_type(dtype)
    itemsize = targetdata.abi_size(lldtype)
    # Compute strides
    rstrides = [itemsize]
    for i, lastsize in enumerate(reversed(shape[1:])):
        rstrides.append(lastsize * rstrides[-1])
    strides = [s for s in reversed(rstrides)]

    kshape = [context.get_constant(types.intp, s) for s in shape]
    kstrides = [context.get_constant(types.intp, s) for s in strides]

    ary.shape = cgutils.pack_array(builder, kshape)
    ary.strides = cgutils.pack_array(builder, kstrides)

    return ary._getvalue()
