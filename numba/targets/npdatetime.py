"""
Implementation of operations on numpy timedelta64.
"""

from llvm.core import Type, Constant
import llvm.core as lc

from numba import npdatetime, types, typing, cgutils, utils
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iterator_impl, iternext_impl,
                                    struct_factory, type_factory)
from numba.typing import signature


TIMEDELTA64 = Type.int(64)
NAT = Constant.int(TIMEDELTA64, npdatetime.NAT)


@type_factory(types.NPTimedelta)
def llvm_timedelta_type(context, tp):
    return TIMEDELTA64


TIMEDELTA_BINOP_SIG = (types.Kind(types.NPTimedelta),) * 2

def scale_timedelta(context, builder, val, srcty, destty):
    """
    Scale the timedelta64 *val* from *srcty* to *destty*
    (both numba.types.NPTimedelta instances)
    """
    factor = npdatetime.get_timedelta_conversion_factor(srcty.unit, destty.unit)
    return builder.mul(Constant.int(TIMEDELTA64, factor), val)

def alloc_timedelta_result(builder, name='ret'):
    """
    Allocate a NaT-initialized timedelta64 result slot.
    """
    ret = cgutils.alloca_once(builder, TIMEDELTA64, name)
    builder.store(NAT, ret)
    return ret

def is_not_nat(builder, val):
    return builder.icmp(lc.ICMP_NE, val, NAT)

def are_not_nat(builder, vals):
    assert len(vals) >= 1
    pred = is_not_nat(builder, vals[0])
    for val in vals[1:]:
        pred = builder.and_(pred, is_not_nat(builder, val))
    return pred


@builtin
@implement('+', *TIMEDELTA_BINOP_SIG)
def timedelta_add_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, are_not_nat(builder, [va, vb])):
        va = scale_timedelta(context, builder, va, ta, sig.return_type)
        vb = scale_timedelta(context, builder, vb, tb, sig.return_type)
        builder.store(builder.add(va, vb), ret)
    return builder.load(ret)

@builtin
@implement('-', *TIMEDELTA_BINOP_SIG)
def timedelta_add_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, are_not_nat(builder, [va, vb])):
        va = scale_timedelta(context, builder, va, ta, sig.return_type)
        vb = scale_timedelta(context, builder, vb, tb, sig.return_type)
        builder.store(builder.sub(va, vb), ret)
    return builder.load(ret)

