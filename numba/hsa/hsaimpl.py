from __future__ import print_function, absolute_import, division
from llvmlite.llvmpy.core import Type
from numba.targets.imputils import implement, Registry
from numba import cgutils
from numba import types
from numba.itanium_mangler import mangle_c
from . import target
from . import stubs

registry = Registry()
register = registry.register

# -----------------------------------------------------------------------------


def _declare_function(context, builder, name, sig, cargs):
    mod = cgutils.get_module(builder)
    llretty = context.get_value_type(sig.return_type)
    llargs = [context.get_value_type(t) for t in sig.args]
    fnty = Type.function(llretty, llargs)
    mangled = mangle_c(name, cargs)
    fn = mod.get_or_insert_function(fnty, mangled)
    fn.calling_convention = target.CC_SPIR_FUNC
    return fn


@register
@implement(stubs.get_global_id, types.uint32)
def get_global_id_impl(context, builder, sig, args):
    [dim] = args
    get_global_id = _declare_function(context, builder, 'get_global_id', sig,
                                      ['unsigned int'])
    return builder.call(get_global_id, [dim])


@register
@implement(stubs.get_local_id, types.uint32)
def get_local_id_impl(context, builder, sig, args):
    [dim] = args
    get_local_id = _declare_function(context, builder, 'get_local_id', sig,
                                     ['unsigned int'])
    return builder.call(get_local_id, [dim])


@register
@implement(stubs.get_global_size, types.uint32)
def get_global_size_impl(context, builder, sig, args):
    [dim] = args
    get_global_size = _declare_function(context, builder, 'get_global_size',
                                        sig, ['unsigned int'])
    return builder.call(get_global_size, [dim])


@register
@implement(stubs.get_local_size, types.uint32)
def get_local_size_impl(context, builder, sig, args):
    [dim] = args
    get_local_size = _declare_function(context, builder, 'get_local_size',
                                       sig, ['unsigned int'])
    return builder.call(get_local_size, [dim])
