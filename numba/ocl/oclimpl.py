from __future__ import print_function, absolute_import, division

import operator
from functools import reduce

from llvmlite.llvmpy.core import Type
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba.targets.imputils import Registry
from numba import cgutils
from numba import types
from numba.itanium_mangler import mangle_c, mangle, mangle_type
from . import target
from . import stubs
from . import enums

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


@lower(stubs.sub_group_barrier)
def sub_group_barrier_impl(context, builder, sig, args):
    assert not args
    barrier = _declare_function(context, builder, 'sub_group_barrier', sig,
                                ['unsigned int'])
    flags = context.get_constant(types.uint32, enums.CLK_LOCAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value
