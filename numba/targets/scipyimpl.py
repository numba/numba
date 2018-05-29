from __future__ import print_function, absolute_import, division

from numba import types, cgutils
from numba.targets.imputils import impl_ret_untracked, Registry


registry = Registry()
lower_getattr = registry.lower_getattr


@lower_getattr(types.LowLevelCallable, 'function')
def llc_function(context, builder, typ, value):
    elems = cgutils.unpack_tuple(builder, value, len(typ))
    res = elems[1]
    return impl_ret_untracked(context, builder, typ, res)
