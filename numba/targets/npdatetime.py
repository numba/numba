"""
Implementation of operations on numpy timedelta64.
"""

from llvmlite.llvmpy.core import Type, Constant
import llvmlite.llvmpy.core as lc

from numba import npdatetime, types, typing, cgutils, utils
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iterator_impl, iternext_impl,
                                    struct_factory, type_factory)
from numba.typing import signature


if not npdatetime.NPDATETIME_SUPPORTED:
    raise NotImplementedError("numpy.datetime64 unsupported in this configuration")


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
    if factor is None:
        # This can happen when using explicit output in a ufunc.
        raise NotImplementedError("cannot convert timedelta64 from %r to %r"
                                  % (srcty.unit, destty.unit))
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
    ret = cgutils.alloca_once(builder, TIMEDELTA64, name=name)
    builder.store(NAT, ret)
    return ret

def alloc_boolean_result(builder, name='ret'):
    """
    Allocate an uninitialized boolean result slot.
    """
    ret = cgutils.alloca_once(builder, Type.int(1), name=name)
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
def timedelta_neg_impl(context, builder, sig, args):
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
@implement(types.sign_type, types.Kind(types.NPTimedelta))
def timedelta_sign_impl(context, builder, sig, args):
    val, = args
    ret = alloc_timedelta_result(builder)
    zero = Constant.int(TIMEDELTA64, 0)
    with cgutils.ifelse(builder, builder.icmp(lc.ICMP_SGT, val, zero)
                        ) as (gt_zero, le_zero):
        with gt_zero:
            builder.store(Constant.int(TIMEDELTA64, 1), ret)
        with le_zero:
            with cgutils.ifelse(builder, builder.icmp(lc.ICMP_EQ, val, zero)
                                ) as (eq_zero, lt_zero):
                with eq_zero:
                    builder.store(Constant.int(TIMEDELTA64, 0), ret)
                with lt_zero:
                    builder.store(Constant.int(TIMEDELTA64, -1), ret)
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


def _timedelta_times_number(context, builder, td_arg, td_type,
                            number_arg, number_type, return_type):
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, is_not_nat(builder, td_arg)):
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fmul(val, number_arg)
            val = builder.fptosi(val, TIMEDELTA64)
        else:
            val = builder.mul(td_arg, number_arg)
        # The scaling is required for ufunc np.multiply() with an explicit
        # output in a different unit.
        val = scale_timedelta(context, builder, val, td_type, return_type)
        builder.store(val, ret)
    return builder.load(ret)

@builtin
@implement('*', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('*', types.Kind(types.NPTimedelta), types.Kind(types.Float))
def timedelta_times_number(context, builder, sig, args):
    return _timedelta_times_number(context, builder,
                                   args[0], sig.args[0], args[1], sig.args[1],
                                   sig.return_type)

@builtin
@implement('*', types.Kind(types.Integer), types.Kind(types.NPTimedelta))
@implement('*', types.Kind(types.Float), types.Kind(types.NPTimedelta))
def number_times_timedelta(context, builder, sig, args):
    return _timedelta_times_number(context, builder,
                                   args[1], sig.args[1], args[0], sig.args[0],
                                   sig.return_type)

@builtin
@implement('/', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('//', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('/?', types.Kind(types.NPTimedelta), types.Kind(types.Integer))
@implement('/', types.Kind(types.NPTimedelta), types.Kind(types.Float))
@implement('//', types.Kind(types.NPTimedelta), types.Kind(types.Float))
@implement('/?', types.Kind(types.NPTimedelta), types.Kind(types.Float))
def timedelta_over_number(context, builder, sig, args):
    td_arg, number_arg = args
    number_type = sig.args[1]
    ret = alloc_timedelta_result(builder)
    ok = builder.and_(is_not_nat(builder, td_arg),
                      builder.not_(cgutils.is_scalar_zero_or_nan(builder, number_arg)))
    with cgutils.if_likely(builder, ok):
        # Denominator is non-zero, non-NaN
        if isinstance(number_type, types.Float):
            val = builder.sitofp(td_arg, number_arg.type)
            val = builder.fdiv(val, number_arg)
            val = builder.fptosi(val, TIMEDELTA64)
        else:
            val = builder.sdiv(td_arg, number_arg)
        # The scaling is required for ufuncs np.*divide() with an explicit
        # output in a different unit.
        val = scale_timedelta(context, builder, val, sig.args[0], sig.return_type)
        builder.store(val, ret)
    return builder.load(ret)

@builtin
@implement('/', *TIMEDELTA_BINOP_SIG)
@implement('/?', *TIMEDELTA_BINOP_SIG)
def timedelta_over_timedelta(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    not_nan = are_not_nat(builder, [va, vb])
    ll_ret_type = context.get_value_type(sig.return_type)
    ret = cgutils.alloca_once(builder, ll_ret_type, name='ret')
    builder.store(Constant.real(ll_ret_type, float('nan')), ret)
    with cgutils.if_likely(builder, not_nan):
        va, vb = normalize_timedeltas(context, builder, va, vb, ta, tb)
        va = builder.sitofp(va, ll_ret_type)
        vb = builder.sitofp(vb, ll_ret_type)
        builder.store(builder.fdiv(va, vb), ret)
    return builder.load(ret)


# Comparison operators on timedelta64

def _create_timedelta_comparison_impl(ll_op, default_value):
    def impl(context, builder, sig, args):
        [va, vb] = args
        [ta, tb] = sig.args
        ret = alloc_boolean_result(builder)
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

    return impl


def _create_timedelta_ordering_impl(ll_op):
    def impl(context, builder, sig, args):
        [va, vb] = args
        [ta, tb] = sig.args
        ret = alloc_boolean_result(builder)
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

    return impl


timedelta_eq_timedelta_impl = _create_timedelta_comparison_impl(lc.ICMP_EQ, cgutils.false_bit)
timedelta_ne_timedelta_impl = _create_timedelta_comparison_impl(lc.ICMP_NE, cgutils.true_bit)
timedelta_lt_timedelta_impl = _create_timedelta_ordering_impl(lc.ICMP_SLT)
timedelta_le_timedelta_impl = _create_timedelta_ordering_impl(lc.ICMP_SLE)
timedelta_gt_timedelta_impl = _create_timedelta_ordering_impl(lc.ICMP_SGT)
timedelta_ge_timedelta_impl = _create_timedelta_ordering_impl(lc.ICMP_SGE)

for op, func in [('==', timedelta_eq_timedelta_impl),
                 ('!=', timedelta_ne_timedelta_impl),
                 ('<',  timedelta_lt_timedelta_impl),
                 ('<=', timedelta_le_timedelta_impl),
                 ('>',  timedelta_gt_timedelta_impl),
                 ('>=', timedelta_ge_timedelta_impl)]:
    builtin(implement(op, *TIMEDELTA_BINOP_SIG)(func))


# Arithmetic on datetime64

def is_leap_year(builder, year_val):
    """
    Return a predicate indicating whether *year_val* (offset by 1970) is a
    leap year.
    """
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
            # NOTE `year_val` is negative, and so will be `from_1972` and `from_2000`.
            # 1972 is the closest later year after 1970.
            # Include the current year, so subtract 2.
            from_1972 = add_constant(builder, year_val, -2)
            # Subtract one day for each 4 years (`from_1972` is negative)
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
        leap_array = cgutils.global_constant(builder, "leap_year_months_acc",
                                             leap_year_months_acc)
        normal_array = cgutils.global_constant(builder, "normal_year_months_acc",
                                               normal_year_months_acc)

        days = cgutils.alloca_once(builder, TIMEDELTA64)

        # First compute year number and month number
        year, month = cgutils.divmod_by_constant(builder, dt_val, 12)

        # Then deduce the number of days
        with cgutils.ifelse(builder,
                            is_leap_year(builder, year)) as (then, otherwise):
            with then:
                addend = builder.load(cgutils.gep(builder, leap_array,
                                                  0, month))
                builder.store(addend, days)
            with otherwise:
                addend = builder.load(cgutils.gep(builder, normal_array,
                                                  0, month))
                builder.store(addend, days)

        days_val = year_to_days(builder, year)
        days_val = builder.add(days_val, builder.load(days))

    if dest_unit_code == 2:
        # Need to scale back to weeks
        weeks, _ = cgutils.divmod_by_constant(builder, days_val, 7)
        return weeks, 'W'
    else:
        return days_val, 'D'


def convert_datetime_for_arith(builder, dt_val, src_unit, dest_unit):
    """
    Convert datetime *dt_val* from *src_unit* to *dest_unit*.
    """
    # First partial conversion to days or weeks, if necessary.
    dt_val, dt_unit = reduce_datetime_for_unit(builder, dt_val, src_unit, dest_unit)
    # Then multiply by the remaining constant factor.
    dt_factor = npdatetime.get_timedelta_conversion_factor(dt_unit, dest_unit)
    if dt_factor is None:
        # This can happen when using explicit output in a ufunc.
        raise NotImplementedError("cannot convert datetime64 from %r to %r"
                                  % (src_unit, dest_unit))
    return scale_by_constant(builder, dt_val, dt_factor)


def _datetime_timedelta_arith(ll_op_name):
    def impl(context, builder, dt_arg, dt_unit,
             td_arg, td_unit, ret_unit):
        ret = alloc_timedelta_result(builder)
        with cgutils.if_likely(builder, are_not_nat(builder, [dt_arg, td_arg])):
            dt_arg = convert_datetime_for_arith(builder, dt_arg,
                                                dt_unit, ret_unit)
            td_factor = npdatetime.get_timedelta_conversion_factor(td_unit, ret_unit)
            td_arg = scale_by_constant(builder, td_arg, td_factor)
            ret_val = getattr(builder, ll_op_name)(dt_arg, td_arg)
            builder.store(ret_val, ret)
        return builder.load(ret)
    return impl

_datetime_plus_timedelta = _datetime_timedelta_arith('add')
_datetime_minus_timedelta = _datetime_timedelta_arith('sub')

# datetime64 + timedelta64

@builtin
@implement('+', types.Kind(types.NPDatetime), types.Kind(types.NPTimedelta))
def datetime_plus_timedelta(context, builder, sig, args):
    dt_arg, td_arg = args
    dt_type, td_type = sig.args
    return _datetime_plus_timedelta(context, builder,
                                    dt_arg, dt_type.unit,
                                    td_arg, td_type.unit,
                                    sig.return_type.unit)

@builtin
@implement('+', types.Kind(types.NPTimedelta), types.Kind(types.NPDatetime))
def timedelta_plus_datetime(context, builder, sig, args):
    td_arg, dt_arg = args
    td_type, dt_type = sig.args
    return _datetime_plus_timedelta(context, builder,
                                    dt_arg, dt_type.unit,
                                    td_arg, td_type.unit,
                                    sig.return_type.unit)

# datetime64 - timedelta64

@builtin
@implement('-', types.Kind(types.NPDatetime), types.Kind(types.NPTimedelta))
def datetime_minus_timedelta(context, builder, sig, args):
    dt_arg, td_arg = args
    dt_type, td_type = sig.args
    return _datetime_minus_timedelta(context, builder,
                                     dt_arg, dt_type.unit,
                                     td_arg, td_type.unit,
                                     sig.return_type.unit)

# datetime64 - datetime64

@builtin
@implement('-', types.Kind(types.NPDatetime), types.Kind(types.NPDatetime))
def datetime_minus_datetime(context, builder, sig, args):
    va, vb = args
    ta, tb = sig.args
    unit_a = ta.unit
    unit_b = tb.unit
    ret_unit = sig.return_type.unit
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, are_not_nat(builder, [va, vb])):
        va = convert_datetime_for_arith(builder, va, unit_a, ret_unit)
        vb = convert_datetime_for_arith(builder, vb, unit_b, ret_unit)
        ret_val = builder.sub(va, vb)
        builder.store(ret_val, ret)
    return builder.load(ret)

# datetime64 comparisons

def _create_datetime_comparison_impl(ll_op):
    def impl(context, builder, sig, args):
        va, vb = args
        ta, tb = sig.args
        unit_a = ta.unit
        unit_b = tb.unit
        ret_unit = npdatetime.get_best_unit(unit_a, unit_b)
        ret = alloc_boolean_result(builder)
        with cgutils.ifelse(builder,
                            are_not_nat(builder, [va, vb])) as (then, otherwise):
            with then:
                norm_a = convert_datetime_for_arith(builder, va, unit_a, ret_unit)
                norm_b = convert_datetime_for_arith(builder, vb, unit_b, ret_unit)
                ret_val = builder.icmp(ll_op, norm_a, norm_b)
                builder.store(ret_val, ret)
            with otherwise:
                # No scaling when comparing NaTs
                ret_val = builder.icmp(ll_op, va, vb)
                builder.store(ret_val, ret)
        return builder.load(ret)

    return impl


datetime_eq_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_EQ)
datetime_ne_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_NE)
datetime_lt_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_SLT)
datetime_le_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_SLE)
datetime_gt_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_SGT)
datetime_ge_datetime_impl = _create_datetime_comparison_impl(lc.ICMP_SGE)

for op, func in [('==', datetime_eq_datetime_impl),
                 ('!=', datetime_ne_datetime_impl),
                 ('<', datetime_lt_datetime_impl),
                 ('<=', datetime_le_datetime_impl),
                 ('>', datetime_gt_datetime_impl),
                 ('>=', datetime_ge_datetime_impl)]:
    builtin(implement(op, *[types.Kind(types.NPDatetime)]*2)(func))


########################################################################
# datetime/timedelta fmax/fmin maximum/minimum support

def datetime_max_impl(context, builder, sig, args):
    # just a regular int64 max avoiding nats.
    # note this could be optimizing relying on the actual value of NAT
    # but as NumPy doesn't rely on this, this seems more resilient
    in1, in2 = args
    in1_not_nat = is_not_nat(builder, in1)
    in2_not_nat = is_not_nat(builder, in2)
    in1_ge_in2 = builder.icmp(lc.ICMP_SGE, in1, in2)
    res = builder.select(in1_ge_in2, in1, in2)
    res = builder.select(in1_not_nat, res, in2)
    res = builder.select(in2_not_nat, res, in1)

    return res


def datetime_min_impl(context, builder, sig, args):
    # just a regular int64 min avoiding nats.
    # note this could be optimizing relying on the actual value of NAT
    # but as NumPy doesn't rely on this, this seems more resilient
    in1, in2 = args
    in1_not_nat = is_not_nat(builder, in1)
    in2_not_nat = is_not_nat(builder, in2)
    in1_le_in2 = builder.icmp(lc.ICMP_SLE, in1, in2)
    res = builder.select(in1_le_in2, in1, in2)
    res = builder.select(in1_not_nat, res, in2)
    res = builder.select(in2_not_nat, res, in1)

    return res


def timedelta_max_impl(context, builder, sig, args):
    # just a regular int64 max avoiding nats.
    # note this could be optimizing relying on the actual value of NAT
    # but as NumPy doesn't rely on this, this seems more resilient
    in1, in2 = args
    in1_not_nat = is_not_nat(builder, in1)
    in2_not_nat = is_not_nat(builder, in2)
    in1_ge_in2 = builder.icmp(lc.ICMP_SGE, in1, in2)
    res = builder.select(in1_ge_in2, in1, in2)
    res = builder.select(in1_not_nat, res, in2)
    res = builder.select(in2_not_nat, res, in1)

    return res


def timedelta_min_impl(context, builder, sig, args):
    # just a regular int64 min avoiding nats.
    # note this could be optimizing relying on the actual value of NAT
    # but as NumPy doesn't rely on this, this seems more resilient
    in1, in2 = args
    in1_not_nat = is_not_nat(builder, in1)
    in2_not_nat = is_not_nat(builder, in2)
    in1_le_in2 = builder.icmp(lc.ICMP_SLE, in1, in2)
    res = builder.select(in1_le_in2, in1, in2)
    res = builder.select(in1_not_nat, res, in2)
    res = builder.select(in2_not_nat, res, in1)

    return res
