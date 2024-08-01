import itertools
from .typeconv import TypeManager, TypeCastingRules
from numba.core import types, config


default_type_manager = TypeManager()


def dump_number_rules():
    tm = default_type_manager
    for a, b in itertools.product(types.number_domain, types.number_domain):
        print(a, '->', b, tm.check_compatible(a, b))


if config.USE_LEGACY_TYPE_SYSTEM: # Old type system
    def _init_casting_rules(tm):
        tcr = TypeCastingRules(tm)
        tcr.safe_unsafe(types.boolean, types.int8)
        tcr.safe_unsafe(types.boolean, types.uint8)

        tcr.promote_unsafe(types.int8, types.int16)
        tcr.promote_unsafe(types.uint8, types.uint16)

        tcr.promote_unsafe(types.int16, types.int32)
        tcr.promote_unsafe(types.uint16, types.uint32)

        tcr.promote_unsafe(types.int32, types.int64)
        tcr.promote_unsafe(types.uint32, types.uint64)

        tcr.safe_unsafe(types.uint8, types.int16)
        tcr.safe_unsafe(types.uint16, types.int32)
        tcr.safe_unsafe(types.uint32, types.int64)

        tcr.safe_unsafe(types.int8, types.float16)
        tcr.safe_unsafe(types.int16, types.float32)
        tcr.safe_unsafe(types.int32, types.float64)


        tcr.unsafe_unsafe(types.int16, types.float16)
        tcr.unsafe_unsafe(types.int32, types.float32)
        # XXX this is inconsistent with the above; but we want to prefer
        # float64 over int64 when typing a heterogeneous operation,
        # e.g. `float64 + int64`.  Perhaps we need more granularity in the
        # conversion kinds.
        tcr.safe_unsafe(types.int64, types.float64)
        tcr.safe_unsafe(types.uint64, types.float64)

        tcr.promote_unsafe(types.float16, types.float32)
        tcr.promote_unsafe(types.float32, types.float64)

        tcr.safe(types.float32, types.complex64)
        tcr.safe(types.float64, types.complex128)

        tcr.promote_unsafe(types.complex64, types.complex128)

        # Allow integers to cast ot void*
        tcr.unsafe_unsafe(types.uintp, types.voidptr)

        return tcr
else: # New type system
    def _init_casting_rules(tm):
        tcr = TypeCastingRules(tm)
        # Python to NumPy conversions
        tcr.safe(types.py_bool, types.np_bool_)
        tcr.safe(types.py_int64, types.np_int64)
        tcr.safe(types.py_float64, types.np_float64)
        tcr.safe(types.py_complex128, types.np_complex128)

        # Pure Python typecasting
        tcr.promote_unsafe(types.py_bool, types.py_intp)
        tcr.promote_unsafe(types.py_int64, types.py_float64)
        tcr.promote_unsafe(types.py_float64, types.py_complex128)

        # Pure NumPy typecasting
        tcr.promote_unsafe(types.np_bool_, types.np_int8)
        tcr.promote_unsafe(types.np_bool_, types.np_uint8)

        tcr.promote_unsafe(types.np_int8, types.np_int16)
        tcr.promote_unsafe(types.np_uint8, types.np_uint16)

        tcr.promote_unsafe(types.np_int16, types.np_int32)
        tcr.promote_unsafe(types.np_uint16, types.np_uint32)

        tcr.promote_unsafe(types.np_int32, types.np_int64)
        tcr.promote_unsafe(types.np_uint32, types.np_uint64)

        tcr.safe_unsafe(types.np_uint8, types.np_int16)
        tcr.safe_unsafe(types.np_uint16, types.np_int32)
        tcr.safe_unsafe(types.np_uint32, types.np_int64)

        tcr.safe_unsafe(types.np_int8, types.np_float16)
        tcr.safe_unsafe(types.np_int16, types.np_float32)
        tcr.safe_unsafe(types.np_int32, types.np_float64)

        tcr.unsafe_unsafe(types.np_int16, types.np_float16)
        tcr.unsafe_unsafe(types.np_int32, types.np_float32)
        # XXX this is inconsistent with the above; but we want to prefer
        # float64 over int64 when typing a heterogeneous operation,
        # e.g. `float64 + int64`.  Perhaps we need more granularity in the
        # conversion kinds.
        tcr.safe_unsafe(types.np_int64, types.np_float64)
        tcr.safe_unsafe(types.np_uint64, types.np_float64)

        tcr.promote_unsafe(types.np_float16, types.np_float32)
        tcr.promote_unsafe(types.np_float32, types.np_float64)

        tcr.safe(types.np_float32, types.np_complex64)
        tcr.safe(types.np_float64, types.np_complex128)

        tcr.promote_unsafe(types.np_complex64, types.np_complex128)

        # Allow integers to cast ot void*
        tcr.unsafe_unsafe(types.np_uintp, types.voidptr)

        return tcr

default_casting_rules = _init_casting_rules(default_type_manager)

