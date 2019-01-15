"""
Helpers to see the refcount information of an object
"""

from numba import types
from numba import cgutils
from numba.extending import intrinsic

from numba.runtime.nrtdynmod import _meminfo_struct_type


def _get_meminfo(context, builder, ty, val):
    datamodel = context.data_model_manager[ty]
    members = datamodel.traverse(builder)

    meminfos = []
    if datamodel.has_nrt_meminfo():
        mi = datamodel.get_nrt_meminfo(builder, val)
        meminfos.append(mi)

    for mtyp, getter in members:
        field = getter(val)
        inner_meminfos = _get_meminfo(context, builder, mtyp, field)
        meminfos.extend(inner_meminfos)
    return meminfos


@intrinsic
def dump_refcount(typingctx, obj):
    """Dump the refcount of an object to stdout.

    Returns True if and only if object is reference-counted and NRT is enabled.
    """
    def codegen(context, builder, signature, args):
        [obj] = args
        [ty] = signature.args

        meminfos = []
        if context.enable_nrt:
            meminfos.extend(_get_meminfo(context, builder, ty, obj))

        if meminfos:
            cgutils.printf(builder, "dump {}".format(ty))
            for mi in meminfos:
                miptr = builder.bitcast(mi, _meminfo_struct_type.as_pointer())
                refctptr = cgutils.gep_inbounds(builder, miptr, 0, 0)
                refct = builder.load(refctptr)
                cgutils.printf(builder, " refct %zd", refct)
            cgutils.printf(builder, ";\n")
            return cgutils.true_bit
        else:
            return cgutils.false_bit

    sig = types.bool_(obj)
    return sig, codegen
