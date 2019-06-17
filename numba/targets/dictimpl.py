"""
This file implements the lowering for `dict()`
"""
from numba.targets.imputils import lower_builtin


@lower_builtin(dict)
def impl_dict(context, builder, sig, args):
    """
    The `dict()` implementation simply forwards the work to `Dict.empty()`.
    """
    from numba.typed import Dict

    dicttype = sig.return_type
    kt, vt = dicttype.key_type, dicttype.value_type

    def call_ctor():
        return Dict.empty(kt, vt)

    return context.compile_internal(builder, call_ctor, sig, args)
