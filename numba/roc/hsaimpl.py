import operator
from functools import reduce

from llvmlite.llvmpy.core import Type
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll
from llvmlite import ir

from numba.core.imputils import Registry
from numba.core import types, cgutils
from numba.core.itanium_mangler import mangle_c, mangle, mangle_type
from numba.roc import target
from numba.roc import stubs
from numba.roc import hlc
from numba.roc import enums

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
    flags = context.get_constant(types.uint32, enums.CLK_GLOBAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value


@lower(stubs.mem_fence, types.uint32)
def mem_fence_impl(context, builder, sig, args):
    [flags] = args
    mem_fence = _declare_function(context, builder, 'mem_fence', sig,
                                ['unsigned int'])
    builder.call(mem_fence, [flags])
    return _void_value


@lower(stubs.wavebarrier)
def wavebarrier_impl(context, builder, sig, args):
    assert not args
    fnty = Type.function(Type.void(), [])
    fn = builder.module.declare_intrinsic('llvm.amdgcn.wave.barrier', fnty=fnty)
    builder.call(fn, [])
    return _void_value

@lower(stubs.activelanepermute_wavewidth,
           types.Any, types.uint32, types.Any, types.bool_)
def activelanepermute_wavewidth_impl(context, builder, sig, args):
    [src, laneid, identity, use_ident] = args
    assert sig.args[0] == sig.args[2]
    elem_type = sig.args[0]
    bitwidth = elem_type.bitwidth
    intbitwidth = Type.int(bitwidth)
    i32 = Type.int(32)
    i1 = Type.int(1)
    name = "__hsail_activelanepermute_wavewidth_b{0}".format(bitwidth)

    fnty = Type.function(intbitwidth, [intbitwidth, i32, intbitwidth, i1])
    fn = builder.module.get_or_insert_function(fnty, name=name)
    fn.calling_convention = target.CC_SPIR_FUNC

    def cast(val):
        return builder.bitcast(val, intbitwidth)

    result = builder.call(fn, [cast(src), laneid, cast(identity), use_ident])
    return builder.bitcast(result, context.get_value_type(elem_type))

def _gen_ds_permute(intrinsic_name):
    def _impl(context, builder, sig, args):
        """
        args are (index, src)
        """
        assert sig.return_type == sig.args[1]
        idx, src = args
        i32 = Type.int(32)
        fnty = Type.function(i32, [i32, i32])
        fn = builder.module.declare_intrinsic(intrinsic_name, fnty=fnty)
        # the args are byte addressable, VGPRs are 4 wide so mul idx by 4
        # the idx might be an int64, this is ok to trunc to int32 as
        # wavefront_size is never likely overflow an int32
        idx = builder.trunc(idx, i32)
        four = lc.Constant.int(i32, 4)
        idx = builder.mul(idx, four)
        # bit cast is so float32 works as packed i32, the return casts back
        result = builder.call(fn, (idx, builder.bitcast(src, i32)))
        return builder.bitcast(result, context.get_value_type(sig.return_type))
    return _impl

lower(stubs.ds_permute, types.Any, types.Any)(_gen_ds_permute('llvm.amdgcn.ds.permute'))
lower(stubs.ds_bpermute, types.Any, types.Any)(_gen_ds_permute('llvm.amdgcn.ds.bpermute'))

@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
@lower(stubs.atomic.add, types.Array,
           types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple,
           types.Any)
def hsail_atomic_add_tuple(context, builder, sig, args):
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

    return builder.atomic_rmw("add", ptr, val, ordering='monotonic')


@lower('hsail.smem.alloc', types.UniTuple, types.Any)
def hsail_smem_alloc_array(context, builder, sig, args):
    shape, dtype = args
    return _generic_array(context, builder, shape=shape, dtype=dtype,
                          symbol_name='_hsapy_smem',
                          addrspace=target.SPIR_LOCAL_ADDRSPACE)


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace):
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

        # Convert to generic address-space
        dataptr = context.addrspacecast(builder, gvmem,
                                        target.SPIR_GENERIC_ADDRSPACE)

    else:
        raise NotImplementedError("addrspace {addrspace}".format(**locals()))

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


def _get_target_data(context):
    return ll.create_target_data(hlc.DATALAYOUT[context.address_size])
