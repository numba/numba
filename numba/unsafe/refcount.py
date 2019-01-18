"""
Helpers to see the refcount information of an object
"""

from numba import types
from numba import cgutils
from numba.extending import intrinsic

from numba.runtime.nrtdynmod import _meminfo_struct_type


@intrinsic
def dump_refcount(typingctx, obj):
    """Dump the refcount of an object to stdout.

    Returns True if and only if object is reference-counted and NRT is enabled.
    """
    def codegen(context, builder, signature, args):
        [obj] = args
        [ty] = signature.args
        # A sequence of (type, meminfo)
        meminfos = []
        if context.enable_nrt:
            tmp_mis = context.nrt.get_meminfos(builder, ty, obj)
            meminfos.extend(tmp_mis)

        if meminfos:
            cgutils.printf(builder, "dump refct of {}".format(ty))
            for ty, mi in meminfos:
                miptr = builder.bitcast(mi, _meminfo_struct_type.as_pointer())
                refctptr = cgutils.gep_inbounds(builder, miptr, 0, 0)
                refct = builder.load(refctptr)
                cgutils.printf(builder, "| {} -> %zd".format(ty), refct)
            cgutils.printf(builder, ";\n")
            return cgutils.true_bit
        else:
            return cgutils.false_bit

    sig = types.bool_(obj)
    return sig, codegen
