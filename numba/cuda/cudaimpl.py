from __future__ import print_function, absolute_import, division

from functools import reduce
import operator

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


@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
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
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    elif aryty.dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float64(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@lower(stubs.atomic.add, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple, types.Any)
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
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float32(lmod), (ptr, val))
    elif aryty.dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_add_float64(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('add', ptr, val, 'monotonic')


@lower(stubs.atomic.max, types.Array, types.intp, types.Any)
def ptx_atomic_max_intp(context, builder, sig, args):
    aryty, indty, valty = sig.args
    ary, ind, val = args
    dtype = aryty.dtype

    if dtype != valty:
        raise TypeError("expect %s but got %s" % (dtype, valty))
    if aryty.ndim != 1:
        raise TypeError("indexing %d-D array with 1-D index" % (aryty.ndim,))

    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(builder, aryty, lary, [ind])

    if dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_max_float64(lmod), (ptr, val))
    else:
        raise TypeError('Unimplemented atomic max with %s array' % dtype)


@lower(stubs.atomic.max, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.max, types.Array, types.UniTuple, types.Any)
def ptx_atomic_max_tuple(context, builder, sig, args):
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

    if aryty.dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_max_float64(lmod), (ptr, val))
    else:
        raise TypeError('Unimplemented atomic max with %s array' % dtype)



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
