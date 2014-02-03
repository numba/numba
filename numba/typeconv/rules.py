from __future__ import print_function, absolute_import
import itertools
from .typeconv import TypeManager
from numba import types


def _init_type_manager():
    """Returns a type manager with default rules
    """
    tm = TypeManager()

    grp_signed = (types.int8, types.int16, types.int32, types.int64)
    grp_unsigned = (types.uint8, types.uint16, types.uint32, types.uint64)
    grp_float = (types.float32, types.float64)
    grp_complex = (types.complex64, types.complex128)

    grp_inter = grp_signed + grp_unsigned + grp_float
    grp_all = grp_inter + grp_complex
    groups = grp_signed, grp_unsigned, grp_float, grp_complex

    # First, all ints and floats are inter-convertible
    for a, b in itertools.product(grp_inter, grp_inter):
        tm.set_unsafe_convert(a, b)

    # Other number types can convert to complex
    for a, b in itertools.product(grp_all, grp_complex):
        tm.set_unsafe_convert(a, b)

    # Setup promotion
    for grp in groups:
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                tm.set_promote(a, b)

    # Setup safe conversion from unsigned to signed
    # Allowed if the result can represent the full range
    for a, b in zip(grp_unsigned, grp_signed[1:]):
        tm.set_safe_convert(a, b)

    # All ints less than 53 bits can safely convert to float64
    f64 = types.float64

    for ty in grp_signed[:-1]:
        tm.set_safe_convert(ty, f64)
    for ty in grp_unsigned[:-1]:
        tm.set_safe_convert(ty, f64)

    # Allow implicit convert from int64 to float64
    tm.set_safe_convert(types.int64, types.float64)
    tm.set_safe_convert(types.uint64, types.float64)

    # All ints less than 24 bits can safely convert to float32
    f32 = types.float32
    for ty in grp_signed[:-2]:
        tm.set_safe_convert(ty, f32)
    for ty in grp_unsigned[:-2]:
        tm.set_safe_convert(ty, f32)

    # All numbers can unsafe convert to bool
    boolean = types.boolean
    for ty in grp_all:
        tm.set_unsafe_convert(ty, boolean)

    return tm


default_type_manager = _init_type_manager()


def dump_number_rules():
    tm = default_type_manager
    for a, b in itertools.product(types.number_domain, types.number_domain):
        print(a, '->', b, tm.check_compatible(a, b))
