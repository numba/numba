import itertools
from .typeconv import TypeManager, TypeCastingRules
from numba.core import types


default_type_manager = TypeManager()


def dump_number_rules():
    tm = default_type_manager
    for a, b in itertools.product(types.number_domain, types.number_domain):
        print(a, '->', b, tm.check_compatible(a, b))


def _init_casting_rules(tm):
    tcr = TypeCastingRules(tm)

    # Pure Python typecasting
    tcr.promote_unsafe(types.py_bool, types.py_intp)
    tcr.promote_unsafe(types.py_int64, types.py_float64)
    tcr.promote_unsafe(types.py_float64, types.py_complex128)

    # Pure NumPy typecasting
    # NEP-50 rules derived from chart at: 
    # https://numpy.org/neps/nep-0050-scalar-promotion.html#schema-of-the-new-proposed-promotion-rules

    # tcr.promote_unsafe(types.np_bool_, types.np_int8)
    # tcr.promote_unsafe(types.np_bool_, types.np_uint8)

    # tcr.promote_unsafe(types.np_uint8, types.np_uint16)
    # tcr.safe_unsafe(types.np_uint8, types.np_int16)
    # tcr.safe_unsafe(types.np_uint8, types.np_float16)

    # tcr.promote_unsafe(types.np_int8, types.np_int16)
    # tcr.safe_unsafe(types.np_int8, types.np_float16)

    # tcr.promote_unsafe(types.np_uint16, types.np_uint32)
    # tcr.safe_unsafe(types.np_uint16, types.np_int32)
    # tcr.safe_unsafe(types.np_uint16, types.np_float32)

    # tcr.promote_unsafe(types.np_int16, types.np_int32)
    # tcr.safe_unsafe(types.np_int16, types.np_float32)

    # tcr.promote_unsafe(types.np_float16, types.np_float32)

    # tcr.promote_unsafe(types.np_uint32, types.np_uint64)
    # tcr.safe_unsafe(types.np_uint32, types.np_int64)

    # tcr.promote_unsafe(types.np_int32, types.np_int64)
    # tcr.safe_unsafe(types.np_int32, types.np_float64)

    # tcr.promote_unsafe(types.np_float32, types.np_float64)
    # tcr.safe(types.np_float32, types.np_complex64)

    # tcr.safe_unsafe(types.np_uint64, types.np_float64)

    # tcr.safe_unsafe(types.np_int64, types.np_float64)

    # tcr.safe(types.np_float64, types.np_complex128)

    # tcr.promote_unsafe(types.np_complex64, types.np_complex128)

    # # Allow integers to cast ot void*
    # tcr.unsafe_unsafe(types.np_uintp, types.voidptr)

    return tcr


default_casting_rules = _init_casting_rules(default_type_manager)
