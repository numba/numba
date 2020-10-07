from __future__ import print_function, absolute_import, division

import operator
from functools import reduce

from llvmlite import ir
from llvmlite.llvmpy.core import Type
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba.core.imputils import Registry
from numba.core import cgutils, types
from numba.core.itanium_mangler import mangle_c, mangle, mangle_type
from numba.dppl import target
from . import stubs
from numba.dppl.codegen import SPIR_DATA_LAYOUT


registry = Registry()
lower = registry.lower

_void_value = lc.Constant.null(lc.Type.pointer(lc.Type.int(8)))

# -----------------------------------------------------------------------------


def _declare_function(context, builder, name, sig, cargs,
                      mangler=mangle_c):
    """Insert declaration for a opencl builtin function.
    Uses the Itanium mangler.

    Args
    ----
    context: target context

    builder: llvm builder

    name: str
        symbol name

    sig: signature
        function signature of the symbol being declared

    cargs: sequence of str
        C type names for the arguments

    mangler: a mangler function
        function to use to mangle the symbol

    """
    mod = builder.module
    if sig.return_type == types.void:
        llretty = lc.Type.void()
    else:
        llretty = context.get_value_type(sig.return_type)
    llargs = [context.get_value_type(t) for t in sig.args]
    fnty = Type.function(llretty, llargs)
    mangled = mangler(name, cargs)
    fn = mod.get_or_insert_function(fnty, mangled)
    fn.calling_convention = target.CC_SPIR_FUNC
    return fn

@lower(stubs.get_global_id, types.uint32)
def get_global_id_impl(context, builder, sig, args):
    [dim] = args
    get_global_id = _declare_function(context, builder, 'get_global_id', sig,
                                      ['unsigned int'])
    res = builder.call(get_global_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_local_id, types.uint32)
def get_local_id_impl(context, builder, sig, args):
    [dim] = args
    get_local_id = _declare_function(context, builder, 'get_local_id', sig,
                                     ['unsigned int'])
    res = builder.call(get_local_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_group_id, types.uint32)
def get_group_id_impl(context, builder, sig, args):
    [dim] = args
    get_group_id = _declare_function(context, builder, 'get_group_id', sig,
                                     ['unsigned int'])
    res = builder.call(get_group_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_num_groups, types.uint32)
def get_num_groups_impl(context, builder, sig, args):
    [dim] = args
    get_num_groups = _declare_function(context, builder, 'get_num_groups', sig,
                                       ['unsigned int'])
    res = builder.call(get_num_groups, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_work_dim)
def get_work_dim_impl(context, builder, sig, args):
    get_work_dim = _declare_function(context, builder, 'get_work_dim', sig,
                                     ["void"])
    res = builder.call(get_work_dim, [])
    return res


@lower(stubs.get_global_size, types.uint32)
def get_global_size_impl(context, builder, sig, args):
    [dim] = args
    get_global_size = _declare_function(context, builder, 'get_global_size',
                                        sig, ['unsigned int'])
    res = builder.call(get_global_size, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_local_size, types.uint32)
def get_local_size_impl(context, builder, sig, args):
    [dim] = args
    get_local_size = _declare_function(context, builder, 'get_local_size',
                                       sig, ['unsigned int'])
    res = builder.call(get_local_size, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.barrier, types.uint32)
def barrier_one_arg_impl(context, builder, sig, args):
    [flags] = args
    barrier = _declare_function(context, builder, 'barrier', sig,
                                ['unsigned int'])
    builder.call(barrier, [flags])
    return _void_value

@lower(stubs.barrier)
def barrier_no_arg_impl(context, builder, sig, args):
    assert not args
    sig = types.void(types.uint32)
    barrier = _declare_function(context, builder, 'barrier', sig,
                                ['unsigned int'])
    flags = context.get_constant(types.uint32, stubs.CLK_GLOBAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value


@lower(stubs.mem_fence, types.uint32)
def mem_fence_impl(context, builder, sig, args):
    [flags] = args
    mem_fence = _declare_function(context, builder, 'mem_fence', sig,
                                ['unsigned int'])
    builder.call(mem_fence, [flags])
    return _void_value


@lower(stubs.sub_group_barrier)
def sub_group_barrier_impl(context, builder, sig, args):
    assert not args
    sig = types.void(types.uint32)
    barrier = _declare_function(context, builder, 'barrier', sig,
                                ['unsigned int'])
    flags = context.get_constant(types.uint32, stubs.CLK_LOCAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value


def insert_and_call_atomic_fn(context, builder, sig, fn_type,
                              dtype, ptr, val, addrspace):
    ll_p = None
    name = ""
    if dtype.name == "int32" or dtype.name == "uint32":
        ll_val = ir.IntType(32)
        ll_p = ll_val.as_pointer()
        if fn_type == "add":
            name = "numba_dppl_atomic_add_i32"
        elif fn_type == "sub":
            name = "numba_dppl_atomic_sub_i32"
        else:
            raise TypeError("Operation type is not supported %s" %
                             (fn_type))
    elif dtype.name == "int64" or dtype.name == "uint64":
        # dpctl needs to expose same functions()
        #if device_env.device_support_int64_atomics():
        if True:
            ll_val = ir.IntType(64)
            ll_p = ll_val.as_pointer()
            if fn_type == "add":
                name = "numba_dppl_atomic_add_i64"
            elif fn_type == "sub":
                name = "numba_dppl_atomic_sub_i64"
            else:
                raise TypeError("Operation type is not supported %s" %
                                 (fn_type))
        #else:
        #    raise TypeError("Current device does not support atomic " +
        #                     "operations on 64-bit Integer")
    elif dtype.name == "float32":
        ll_val = ir.FloatType()
        ll_p = ll_val.as_pointer()
        if fn_type == "add":
            name = "numba_dppl_atomic_add_f32"
        elif fn_type == "sub":
            name = "numba_dppl_atomic_sub_f32"
        else:
            raise TypeError("Operation type is not supported %s" %
                             (fn_type))
    elif dtype.name == "float64":
        #if device_env.device_support_float64_atomics():
        # dpctl needs to expose same functions()
        if True:
            ll_val = ir.DoubleType()
            ll_p = ll_val.as_pointer()
            if fn_type == "add":
                name = "numba_dppl_atomic_add_f64"
            elif fn_type == "sub":
                name = "numba_dppl_atomic_sub_f64"
            else:
                raise TypeError("Operation type is not supported %s" %
                                 (fn_type))
        #else:
        #    raise TypeError("Current device does not support atomic " +
        #                    "operations on 64-bit Float")
    else:
        raise TypeError("Atomic operation is not supported for type %s" %
                        (dtype.name))

    if addrspace == target.SPIR_LOCAL_ADDRSPACE:
        name = name + "_local"
    else:
        name = name + "_global"

    assert(ll_p != None)
    assert(name != "")
    ll_p.addrspace = target.SPIR_GENERIC_ADDRSPACE

    mod = builder.module
    if sig.return_type == types.void:
        llretty = lc.Type.void()
    else:
        llretty = context.get_value_type(sig.return_type)

    llargs = [ll_p, context.get_value_type(sig.args[2])]
    fnty = ir.FunctionType(llretty, llargs)

    fn = mod.get_or_insert_function(fnty, name)
    fn.calling_convention = target.CC_SPIR_FUNC

    generic_ptr = context.addrspacecast(builder, ptr,
                                    target.SPIR_GENERIC_ADDRSPACE)

    return builder.call(fn, [generic_ptr, val])


@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
@lower(stubs.atomic.add, types.Array,
           types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple,
           types.Any)
def atomic_add_tuple(context, builder, sig, args):
    from .atomics import atomic_support_present
    if atomic_support_present():
        context.link_binaries[target.LINK_ATOMIC] = True
        aryty, indty, valty = sig.args
        ary, inds, val = args
        dtype = aryty.dtype

        if indty == types.intp:
            indices = [inds]  # just a single integer
            indty = [indty]
        else:
            indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
            indices = [context.cast(builder, i, t, types.intp)
                       for t, i in zip(indty, indices)]

        if dtype != valty:
            raise TypeError("expecting %s but got %s" % (dtype, valty))

        if aryty.ndim != len(indty):
            raise TypeError("indexing %d-D array with %d-D index" %
                            (aryty.ndim, len(indty)))

        lary = context.make_array(aryty)(context, builder, ary)
        ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices)

        if aryty.addrspace == target.SPIR_LOCAL_ADDRSPACE:
            return insert_and_call_atomic_fn(context, builder, sig, "add", dtype,
                    ptr, val, target.SPIR_LOCAL_ADDRSPACE)
        else:
            return insert_and_call_atomic_fn(context, builder, sig, "add", dtype,
                    ptr, val, target.SPIR_GLOBAL_ADDRSPACE)
    else:
        raise ImportError("Atomic support is not present, can not perform atomic_add")


@lower(stubs.atomic.sub, types.Array, types.intp, types.Any)
@lower(stubs.atomic.sub, types.Array,
           types.UniTuple, types.Any)
@lower(stubs.atomic.sub, types.Array, types.Tuple,
           types.Any)
def atomic_sub_tuple(context, builder, sig, args):
    from .atomics import atomic_support_present
    if atomic_support_present():
        context.link_binaries[target.LINK_ATOMIC] = True
        aryty, indty, valty = sig.args
        ary, inds, val = args
        dtype = aryty.dtype

        if indty == types.intp:
            indices = [inds]  # just a single integer
            indty = [indty]
        else:
            indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
            indices = [context.cast(builder, i, t, types.intp)
                       for t, i in zip(indty, indices)]

        if dtype != valty:
            raise TypeError("expecting %s but got %s" % (dtype, valty))

        if aryty.ndim != len(indty):
            raise TypeError("indexing %d-D array with %d-D index" %
                            (aryty.ndim, len(indty)))

        lary = context.make_array(aryty)(context, builder, ary)
        ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices)


        if aryty.addrspace == target.SPIR_LOCAL_ADDRSPACE:
            return insert_and_call_atomic_fn(context, builder, sig, "sub", dtype,
                    ptr, val, target.SPIR_LOCAL_ADDRSPACE)
        else:
            return insert_and_call_atomic_fn(context, builder, sig, "sub", dtype,
                    ptr, val, target.SPIR_GLOBAL_ADDRSPACE)
    else:
        raise ImportError("Atomic support is not present, can not perform atomic_add")


@lower('dppl.lmem.alloc', types.UniTuple, types.Any)
def dppl_lmem_alloc_array(context, builder, sig, args):
    shape, dtype = args
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_dppl_lmem',
                          addrspace=target.SPIR_LOCAL_ADDRSPACE)


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace):
    """
    This function allows us to create generic arrays in different
    address spaces.
    """
    elemcount = reduce(operator.mul, shape)
    lldtype = context.get_data_type(dtype)
    laryty = Type.array(lldtype, elemcount)

    if addrspace == target.SPIR_LOCAL_ADDRSPACE:
        lmod = builder.module

        # Create global variable in the requested address-space
        gvmem = lmod.add_global_variable(laryty, symbol_name, addrspace)

        if elemcount <= 0:
            raise ValueError("array length <= 0")
        else:
            gvmem.linkage = lc.LINKAGE_INTERNAL

        if dtype not in types.number_domain:
            raise TypeError("unsupported type: %s" % dtype)

    else:
        raise NotImplementedError("addrspace {addrspace}".format(**locals()))

    # We need to add the addrspace to _make_array() function call as we want
    # the variable containing the reference of the memory to retain the
    # original address space of that memory. Before, we were casting the
    # memories allocated in local address space to global address space. This
    # approach does not let us identify the original address space of a memory
    # down the line.
    return _make_array(context, builder, gvmem, dtype, shape, addrspace=addrspace)


def _make_array(context, builder, dataptr, dtype, shape, layout='C', addrspace=target.SPIR_GENERIC_ADDRSPACE):
    ndim = len(shape)
    # Create array object
    aryty = types.Array(dtype=dtype, ndim=ndim, layout='C', addrspace=addrspace)
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


def _get_target_data(context):
    return ll.create_target_data(SPIR_DATA_LAYOUT[context.address_size])
