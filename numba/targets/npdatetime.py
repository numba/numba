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


# datetime64 and timedelta64 use the same internal representation
DATETIME64 = TIMEDELTA64 = Type.int(64)
NAT = Constant.int(TIMEDELTA64, npdatetime.NAT)

TIMEDELTA_BINOP_SIG = (types.Kind(types.NPTimedelta),) * 2


@type_factory(types.NPDatetime)
def llvm_datetime_type(context, tp):
    return DATETIME64

@type_factory(types.NPTimedelta)
def llvm_timedelta_type(context, tp):
    return TIMEDELTA64


def scale_by_constant(builder, val, factor):
    """
    Multiply *val* by the constant *factor*.
    """
    return builder.mul(val, Constant.int(TIMEDELTA64, factor))

def unscale_by_constant(builder, val, factor):
    """
    Divide *val* by the constant *factor*.
    """
    return builder.sdiv(val, Constant.int(TIMEDELTA64, factor))

def add_constant(builder, val, const):
    """
    Add constant *const* to *val*.
    """
    return builder.add(val, Constant.int(TIMEDELTA64, const))

def scale_timedelta(context, builder, val, srcty, destty):
    """
    Scale the timedelta64 *val* from *srcty* to *destty*
    (both numba.types.NPTimedelta instances)
    """
    factor = npdatetime.get_timedelta_conversion_factor(srcty.unit, destty.unit)
    return scale_by_constant(builder, val, factor)

def normalize_timedeltas(context, builder, left, right, leftty, rightty):
    """
    Scale either *left* or *right* to the other's unit, in order to have
    homogenous units.
    """
    factor = npdatetime.get_timedelta_conversion_factor(leftty.unit, rightty.unit)
    if factor is not None:
        return scale_by_constant(builder, left, factor), right
    factor = npdatetime.get_timedelta_conversion_factor(rightty.unit, leftty.unit)
    if factor is not None:
        return left, scale_by_constant(builder, right, factor)
    # Typing should not let this happen, except on == and != operators
    raise RuntimeError("cannot normalize %r and %r" % (leftty, rightty))

def alloc_timedelta_result(builder, name='ret'):
    """
    Allocate a NaT-initialized datetime64 (or timedelta64) result slot.
    """
    ret = cgutils.alloca_once(builder, TIMEDELTA64, name)
    builder.store(NAT, ret)
    return ret

def is_not_nat(builder, val):
    """
    Return a predicate which is true if *val* is not NaT.
    """
    return builder.icmp(lc.ICMP_NE, val, NAT)

def are_not_nat(builder, vals):
    """
    Return a predicate which is true if all of *vals* are not NaT.
    """
    assert len(vals) >= 1
    pred = is_not_nat(builder, vals[0])
    for val in vals[1:]:
        pred = builder.and_(pred, is_not_nat(builder, val))
    return pred


def make_constant_array(vals):
    consts = [Constant.int(TIMEDELTA64, v) for v in vals]
    return Constant.array(TIMEDELTA64, consts)

normal_year_months = make_constant_array([31, 28, 31, 30, 31, 30,
                                          31, 31, 30, 31, 30, 31])
leap_year_months = make_constant_array([31, 29, 31, 30, 31, 30,
                                        31, 31, 30, 31, 30, 31])
normal_year_months_acc = make_constant_array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
leap_year_months_acc = make_constant_array(
    [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])

# Arithmetic operators on timedelta64

@builtin
@implement('+', types.Kind(types.NPTimedelta))
def timedelta_pos_impl(context, builder, sig, args):
    return args[0]

@builtin
@implement('-', types.Kind(types.NPTimedelta))
def timedelta_pos_impl(context, builder, sig, args):
    return builder.neg(args[0])

@builtin
@implement(types.abs_type, types.Kind(types.NPTimedelta))
def timedelta_abs_impl(context, builder, sig, args):
    val, = args
    ret = alloc_timedelta_result(builder)
    with cgutils.ifelse(builder,
                        cgutils.is_scalar_neg(builder, val)) as (then, otherwise):
        with then:
            builder.store(builder.neg(val), ret)
        with otherwise:
            builder.store(val, ret)
    return builder.load(ret)

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
def timedelta_sub_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, are_not_nat(builder, [va, vb])):
        va = scale_timedelta(context, builder, va, ta, sig.return_type)
        vb = scale_timedelta(context, builder, vb, tb, sig.return_type)
        builder.store(builder.sub(va, vb), ret)
    return builder.load(ret)


def _timedelta_times_number(context, builder, td_arg, number_arg, number_type):
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, is_not_nat(builder, td_arg)):
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fmul(val, number_arg)
            val = builder.fptosi(val, TIMEDELTA64)
        else:
            val = builder.mul(td_arg, number_arg)
        builder.store(val, ret)
    return builder.load(ret)

@builtin
@implement('*', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('*', types.Kind(types.NPTimedelta), types.Kind(types.Float))
def timedelta_mul_impl(context, builder, sig, args):
    return _timedelta_times_number(context, builder,
                                  args[0], args[1], sig.args[1])

@builtin
@implement('*', types.Kind(types.Integer), types.Kind(types.NPTimedelta))
@implement('*', types.Kind(types.Float), types.Kind(types.NPTimedelta))
def timedelta_mul_impl(context, builder, sig, args):
    return _timedelta_times_number(context, builder,
                                  args[1], args[0], sig.args[0])

@builtin
@implement('/', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('//', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('/?', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('/', types.Kind(types.NPTimedelta), types.Kind(types.Float))
@implement('//', types.Kind(types.NPTimedelta), types.Kind(types.Float))
@implement('/?', types.Kind(types.NPTimedelta), types.Kind(types.Float))
def timedelta_div_impl(context, builder, sig, args):
    td_arg, number_arg = args
    number_type = sig.args[1]
    ret = alloc_timedelta_result(builder)
    ok = builder.and_(is_not_nat(builder, td_arg),
                      builder.not_(cgutils.is_scalar_zero(builder, number_arg)))
    with cgutils.if_likely(builder, ok):
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fdiv(val, number_arg)
            val = builder.fptosi(val, TIMEDELTA64)
        else:
            val = builder.sdiv(td_arg, number_arg)
        builder.store(val, ret)
    return builder.load(ret)

@builtin
@implement('/', *TIMEDELTA_BINOP_SIG)
@implement('/?', *TIMEDELTA_BINOP_SIG)
def timedelta_div_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    not_nan = are_not_nat(builder, [va, vb])
    ll_ret_type = context.get_value_type(sig.return_type)
    ret = cgutils.alloca_once(builder, ll_ret_type, 'ret')
    builder.store(Constant.real(ll_ret_type, float('nan')), ret)
    with cgutils.if_likely(builder, not_nan):
        va, vb = normalize_timedeltas(context, builder, va, vb, ta, tb)
        va = builder.sitofp(va, ll_ret_type)
        vb = builder.sitofp(vb, ll_ret_type)
        builder.store(builder.fdiv(va, vb), ret)
    return builder.load(ret)


# Comparison operators on timedelta64

def implement_equality_operator(py_op, ll_op, default_value):
    @builtin
    @implement(py_op, *TIMEDELTA_BINOP_SIG)
    def timedelta_eq_impl(context, builder, sig, args):
        [va, vb] = args
        [ta, tb] = sig.args
        ret = cgutils.alloca_once(builder, Type.int(1), 'ret')
        with cgutils.ifelse(builder, are_not_nat(builder, [va, vb])) as (then, otherwise):
            with then:
                try:
                    norm_a, norm_b = normalize_timedeltas(context, builder, va, vb, ta, tb)
                except RuntimeError:
                    # Cannot normalize units => the values are unequal (except if NaT)
                    builder.store(default_value, ret)
                else:
                    builder.store(builder.icmp(ll_op, norm_a, norm_b), ret)
            with otherwise:
                # No scaling when comparing NaTs
                builder.store(builder.icmp(ll_op, va, vb), ret)
        return builder.load(ret)

implement_equality_operator('==', lc.ICMP_EQ, cgutils.false_bit)
implement_equality_operator('!=', lc.ICMP_NE, cgutils.true_bit)


def implement_ordering_operator(py_op, ll_op):
    @builtin
    @implement(py_op, *TIMEDELTA_BINOP_SIG)
    def timedelta_eq_impl(context, builder, sig, args):
        [va, vb] = args
        [ta, tb] = sig.args
        ret = cgutils.alloca_once(builder, Type.int(1), 'ret')
        with cgutils.ifelse(builder, are_not_nat(builder, [va, vb])) as (then, otherwise):
            with then:
                norm_a, norm_b = normalize_timedeltas(context, builder, va, vb, ta, tb)
                builder.store(builder.icmp(ll_op, norm_a, norm_b), ret)
            with otherwise:
                # No scaling when comparing NaT with something else
                # (i.e. NaT is <= everything else, since it's the smallest
                #  int64 value)
                builder.store(builder.icmp(ll_op, va, vb), ret)
        return builder.load(ret)

implement_ordering_operator('<', lc.ICMP_SLT)
implement_ordering_operator('<=', lc.ICMP_SLE)
implement_ordering_operator('>', lc.ICMP_SGT)
implement_ordering_operator('>=', lc.ICMP_SGE)


# Arithmetic on datetime64

def is_leap_year(builder, year_val):
    actual_year = builder.add(year_val, Constant.int(DATETIME64, 1970))
    multiple_of_4 = cgutils.is_null(
        builder, builder.and_(actual_year, Constant.int(DATETIME64, 3)))
    not_multiple_of_100 = cgutils.is_not_null(
        builder, builder.srem(actual_year, Constant.int(DATETIME64, 100)))
    multiple_of_400 = cgutils.is_null(
        builder, builder.srem(actual_year, Constant.int(DATETIME64, 400)))
    return builder.and_(multiple_of_4,
                        builder.or_(not_multiple_of_100, multiple_of_400))

def year_to_days(builder, year_val):
    """
    Given a year *year_val* (offset to 1970), return the number of days
    since the 1970 epoch.
    """
    # The algorithm below is copied from Numpy's get_datetimestruct_days()
    # (src/multiarray/datetime.c)
    ret = cgutils.alloca_once(builder, TIMEDELTA64)
    # First approximation
    days = scale_by_constant(builder, year_val, 365)
    # Adjust for leap years
    with cgutils.ifelse(builder, cgutils.is_neg_int(builder, year_val)) \
        as (if_neg, if_pos):
        with if_pos:
            # At or after 1970:
            # 1968 is the closest leap year before 1970.
            # Exclude the current year, so add 1.
            from_1968 = add_constant(builder, year_val, 1)
            # Add one day for each 4 years
            p_days = builder.add(days,
                                 unscale_by_constant(builder, from_1968, 4))
            # 1900 is the closest previous year divisible by 100
            from_1900 = add_constant(builder, from_1968, 68)
            # Subtract one day for each 100 years
            p_days = builder.sub(p_days,
                                 unscale_by_constant(builder, from_1900, 100))
            # 1600 is the closest previous year divisible by 400
            from_1600 = add_constant(builder, from_1900, 300)
            # Add one day for each 400 years
            p_days = builder.add(p_days,
                                 unscale_by_constant(builder, from_1600, 400))
            builder.store(p_days, ret)
        with if_neg:
            # Before 1970:
            # 1972 is the closest later year after 1970.
            # Include the current year, so subtract 2.
            from_1972 = add_constant(builder, year_val, -2)
            # Subtract one day for each 4 years
            n_days = builder.add(days,
                                 unscale_by_constant(builder, from_1972, 4))
            # 2000 is the closest later year divisible by 100
            from_2000 = add_constant(builder, from_1972, -28)
            # Add one day for each 100 years
            n_days = builder.sub(n_days,
                                 unscale_by_constant(builder, from_2000, 100))
            # 2000 is also the closest later year divisible by 400
            # Subtract one day for each 400 years
            n_days = builder.add(n_days,
                                 unscale_by_constant(builder, from_2000, 400))
            builder.store(n_days, ret)
    return builder.load(ret)


def reduce_datetime_for_unit(builder, dt_val, src_unit, dest_unit):
    dest_unit_code = npdatetime.DATETIME_UNITS[dest_unit]
    src_unit_code = npdatetime.DATETIME_UNITS[src_unit]
    if dest_unit_code < 2 or src_unit_code >= 2:
        return dt_val, src_unit
    # Need to compute the day ordinal for *dt_val*
    if src_unit_code == 0:
        # Years to days
        year_val = dt_val
        days_val = year_to_days(builder, year_val)

    else:
        # Months to days
        module = cgutils.get_module(builder)

        leap_array = cgutils.global_constant(builder, "leap_year_months_acc",
                                             leap_year_months_acc)
        normal_array = cgutils.global_constant(builder, "normal_year_months_acc",
                                               normal_year_months_acc)

        year = cgutils.alloca_once(builder, TIMEDELTA64)
        month = cgutils.alloca_once(builder, TIMEDELTA64)
        days = cgutils.alloca_once(builder, TIMEDELTA64)

        # First compute year number and month number
        with cgutils.ifelse(builder, cgutils.is_neg_int(builder, dt_val)
                            ) as (if_neg, if_pos):
            with if_pos:
                # year = dt / 12
                year_val = unscale_by_constant(builder, dt_val, 12)
                builder.store(year_val, year)
                # month = dt % 12
                month_val = builder.srem(dt_val,
                                         Constant.int(TIMEDELTA64, 12))
                builder.store(month_val, month)
            with if_neg:
                # Basically, we want Python divmod() semantics but
                # we must deal with C-like signed division.
                # year = -1 + (dt + 1) / 12
                dt_plus_one = add_constant(builder, dt_val, 1)
                year_val = unscale_by_constant(builder, dt_plus_one, 12)
                year_val = add_constant(builder, year_val, -1)
                builder.store(year_val, year)
                # month = 11 + (dt + 1) % 12
                month_val = builder.srem(dt_plus_one,
                                         Constant.int(TIMEDELTA64, 12))
                month_val = add_constant(builder, month_val, 11)
                builder.store(month_val, month)

        year_val = builder.load(year)
        month_val = builder.load(month)

        # Then deduce the number of days
        with cgutils.ifelse(builder,
                            is_leap_year(builder, year_val)) as (then, otherwise):
            with then:
                addend = builder.load(cgutils.gep(builder, leap_array,
                                                  0, month_val))
                builder.store(addend, days)
                #builder.store(builder.add(days_val, addend), days_ret)
            with otherwise:
                addend = builder.load(cgutils.gep(builder, normal_array,
                                                  0, month_val))
                builder.store(addend, days)
                #builder.store(builder.add(days_val, addend), days_ret)

        days_val = year_to_days(builder, year_val)
        days_val = builder.add(days_val, builder.load(days))

    if dest_unit_code == 2:
        # Need to scale back to weeks
        return unscale_by_constant(builder, days_val, 7), 'W'

    return days_val, 'D'


def _datetime_timedelta_arith(ll_op_name):
    def impl(context, builder, dt_arg, dt_unit,
             td_arg, td_unit, ret_unit):
        ret = alloc_timedelta_result(builder)
        with cgutils.if_likely(builder, are_not_nat(builder, [dt_arg, td_arg])):
            dt_arg, dt_unit = reduce_datetime_for_unit(builder, dt_arg, dt_unit, ret_unit)
            dt_factor = npdatetime.get_timedelta_conversion_factor(dt_unit, ret_unit)
            td_factor = npdatetime.get_timedelta_conversion_factor(td_unit, ret_unit)
            dt_arg = scale_by_constant(builder, dt_arg, dt_factor)
            td_arg = scale_by_constant(builder, td_arg, td_factor)
            ret_val = getattr(builder, ll_op_name)(dt_arg, td_arg)
            builder.store(ret_val, ret)
        return builder.load(ret)
    return impl

_datetime_plus_timedelta = _datetime_timedelta_arith('add')
_datetime_minus_timedelta = _datetime_timedelta_arith('sub')

@builtin
@implement('+', types.Kind(types.NPDatetime), types.Kind(types.NPTimedelta))
def timedelta_add_impl(context, builder, sig, args):
    dt_arg, td_arg = args
    dt_type, td_type = sig.args
    return _datetime_plus_timedelta(context, builder,
                                    dt_arg, dt_type.unit,
                                    td_arg, td_type.unit,
                                    sig.return_type.unit)

@builtin
@implement('+', types.Kind(types.NPTimedelta), types.Kind(types.NPDatetime))
def timedelta_add_impl(context, builder, sig, args):
    td_arg, dt_arg = args
    td_type, dt_type = sig.args
    return _datetime_plus_timedelta(context, builder,
                                    dt_arg, dt_type.unit,
                                    td_arg, td_type.unit,
                                    sig.return_type.unit)

@builtin
@implement('-', types.Kind(types.NPDatetime), types.Kind(types.NPTimedelta))
def timedelta_sub_impl(context, builder, sig, args):
    dt_arg, td_arg = args
    dt_type, td_type = sig.args
    return _datetime_minus_timedelta(context, builder,
                                     dt_arg, dt_type.unit,
                                     td_arg, td_type.unit,
                                     sig.return_type.unit)
