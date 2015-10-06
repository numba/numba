from __future__ import print_function, absolute_import, division

import math
from functools import reduce

from llvmlite import ir
from llvmlite.llvmpy.core import Type, Constant
import llvmlite.llvmpy.core as lc

from .imputils import (builtin, builtin_attr, implement, impl_attribute,
                       impl_attribute_generic, iternext_impl,
                       impl_ret_borrowed, impl_ret_untracked)
from . import optional
from .. import typing, types, cgutils, utils, intrinsics


@builtin
@implement('is not', types.Any, types.Any)
def generic_is_not(context, builder, sig, args):
    """
    Implement `x is not y` as `not (x is y)`.
    """
    is_impl = context.get_function('is', sig)
    return builder.not_(is_impl(builder, args))

#-------------------------------------------------------------------------------

def _int_arith_flags(rettype):
    """
    Return the modifier flags for integer arithmetic.
    """
    if rettype.signed:
        # Ignore the effects of signed overflow.  This is important for
        # optimization of some indexing operations.  For example
        # array[i+1] could see `i+1` trigger a signed overflow and
        # give a negative number.  With Python's indexing, a negative
        # index is treated differently: its resolution has a runtime cost.
        # Telling LLVM to ignore signed overflows allows it to optimize
        # away the check for a negative `i+1` if it knows `i` is positive.
        return ['nsw']
    else:
        return []


def int_add_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    res = builder.add(a, b, flags=_int_arith_flags(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sub_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    res = builder.sub(a, b, flags=_int_arith_flags(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_mul_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    res = builder.mul(a, b, flags=_int_arith_flags(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_divmod(context, builder, x, y):
    """
    Reference Objects/intobject.c
    xdivy = x / y;
    xmody = (long)(x - (unsigned long)xdivy * y);
    /* If the signs of x and y differ, and the remainder is non-0,
     * C89 doesn't define whether xdivy is now the floor or the
     * ceiling of the infinitely precise quotient.  We want the floor,
     * and we have it iff the remainder's sign matches y's.
     */
    if (xmody && ((y ^ xmody) < 0) /* i.e. and signs differ */) {
        xmody += y;
        --xdivy;
        assert(xmody && ((y ^ xmody) >= 0));
    }
    *p_xdivy = xdivy;
    *p_xmody = xmody;
    """
    assert x.type == y.type
    xdivy = builder.sdiv(x, y)
    xmody = builder.srem(x, y)  # Intel has divmod instruction

    ZERO = Constant.null(y.type)
    ONE = Constant.int(y.type, 1)

    y_xor_xmody_ltz = builder.icmp(lc.ICMP_SLT, builder.xor(y, xmody), ZERO)
    xmody_istrue = builder.icmp(lc.ICMP_NE, xmody, ZERO)
    cond = builder.and_(xmody_istrue, y_xor_xmody_ltz)

    bb1 = builder.basic_block
    with builder.if_then(cond):
        xmody_plus_y = builder.add(xmody, y)
        xdivy_minus_1 = builder.sub(xdivy, ONE)
        bb2 = builder.basic_block

    resdiv = builder.phi(y.type)
    resdiv.add_incoming(xdivy, bb1)
    resdiv.add_incoming(xdivy_minus_1, bb2)

    resmod = builder.phi(x.type)
    resmod.add_incoming(xmody, bb1)
    resmod.add_incoming(xmody_plus_y, bb2)

    return resdiv, resmod


@builtin
@implement('/?', types.Kind(types.Integer), types.Kind(types.Integer))
@implement('//', types.Kind(types.Integer), types.Kind(types.Integer))
def int_floordiv_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    res = cgutils.alloca_once(builder, a.type)

    with builder.if_else(cgutils.is_scalar_zero(builder, b), likely=False
                         ) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(
                builder, ("integer division by zero",)):
                # No exception raised => return 0
                # XXX We should also set the FPU exception status, but
                # there's no easy way to do that from LLVM.
                builder.store(b, res)
        with if_non_zero:
            if sig.return_type.signed:
                quot, _ = int_divmod(context, builder, a, b)
            else:
                quot = builder.udiv(a, b)
            builder.store(quot, res)

    return impl_ret_untracked(context, builder, sig.return_type,
                              builder.load(res))


@builtin
@implement('/', types.Kind(types.Integer), types.Kind(types.Integer))
def int_truediv_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    with cgutils.if_zero(builder, b):
        context.error_model.fp_zero_division(builder, ("division by zero",))
    res = builder.fdiv(a, b)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement('%', types.Kind(types.Integer), types.Kind(types.Integer))
def int_rem_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    res = cgutils.alloca_once(builder, a.type)

    with builder.if_else(cgutils.is_scalar_zero(builder, b), likely=False
                         ) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(
                builder, ("modulo by zero",)):
                # No exception raised => return 0
                # XXX We should also set the FPU exception status, but
                # there's no easy way to do that from LLVM.
                builder.store(b, res)
        with if_non_zero:
            if sig.return_type.signed:
                _, rem = int_divmod(context, builder, a, b)
            else:
                rem = builder.urem(a, b)
            builder.store(rem, res)

    return impl_ret_untracked(context, builder, sig.return_type,
                              builder.load(res))


def int_power_impl(context, builder, sig, args):
    """
    a ^ b, where a is an integer or real, and b an integer
    """
    is_integer = isinstance(sig.args[0], types.Integer)
    tp = sig.return_type
    zerodiv_return = False
    if is_integer and not context.error_model.raise_on_fp_zero_division:
        # If not raising, return 0x8000... when computing 0 ** <negative number>
        zerodiv_return = -1 << (tp.bitwidth - 1)

    def int_power(a, b):
        # Ensure computations are done with a large enough width
        r = tp(1)
        a = tp(a)
        if b < 0:
            invert = True
            exp = -b
            if exp < 0:
                raise OverflowError
            if is_integer:
                if a == 0:
                    if zerodiv_return:
                        return zerodiv_return
                    else:
                        raise ZeroDivisionError("0 cannot be raised to a negative power")
                if a != 1 and a != -1:
                    return 0
        else:
            invert = False
            exp = b
        if exp > 0x10000:
            # Optimization cutoff: fallback on the generic algorithm
            return math.pow(a, float(b))
        while exp != 0:
            if exp & 1:
                r *= a
            exp >>= 1
            a *= a

        return 1.0 / r if invert else r

    res = context.compile_internal(builder, int_power, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_slt_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_SLT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sle_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_SLE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sgt_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_SGT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sge_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_SGE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ult_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_ULT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ule_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_ULE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ugt_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_UGT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_uge_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_UGE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_eq_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_EQ, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ne_impl(context, builder, sig, args):
    res = builder.icmp(lc.ICMP_NE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_abs_impl(context, builder, sig, args):
    [x] = args
    ZERO = Constant.null(x.type)
    ltz = builder.icmp(lc.ICMP_SLT, x, ZERO)
    negated = builder.neg(x)
    res = builder.select(ltz, negated, x)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def uint_abs_impl(context, builder, sig, args):
    [x] = args
    return impl_ret_untracked(context, builder, sig.return_type, x)


def int_shl_impl(context, builder, sig, args):
    [valty, amtty] = sig.args
    [val, amt] = args
    val = context.cast(builder, val, valty, sig.return_type)
    amt = context.cast(builder, amt, amtty, sig.return_type)
    res = builder.shl(val, amt)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_shr_impl(context, builder, sig, args):
    [valty, amtty] = sig.args
    [val, amt] = args
    val = context.cast(builder, val, valty, sig.return_type)
    amt = context.cast(builder, amt, amtty, sig.return_type)
    if sig.return_type.signed:
        res = builder.ashr(val, amt)
    else:
        res = builder.lshr(val, amt)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_and_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    res = builder.and_(cav, cbc)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_or_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    res = builder.or_(cav, cbc)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_xor_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    res = builder.xor(cav, cbc)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_negate_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    # Negate before upcasting, for unsigned numbers
    res = builder.neg(val)
    res = context.cast(builder, res, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_positive_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    res = context.cast(builder, val, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_invert_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    # Invert before upcasting, for unsigned numbers
    res = builder.xor(val, Constant.all_ones(val.type))
    res = context.cast(builder, res, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def bool_invert_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    res = builder.sub(Constant.int(val.type, 1), val)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sign_impl(context, builder, sig, args):
    [x] = args
    POS = Constant.int(x.type, 1)
    NEG = Constant.int(x.type, -1)
    ZERO = Constant.int(x.type, 0)

    cmp_zero = builder.icmp(lc.ICMP_EQ, x, ZERO)
    cmp_pos = builder.icmp(lc.ICMP_SGT, x, ZERO)

    presult = cgutils.alloca_once(builder, x.type)

    bb_zero = builder.append_basic_block(".zero")
    bb_postest = builder.append_basic_block(".postest")
    bb_pos = builder.append_basic_block(".pos")
    bb_neg = builder.append_basic_block(".neg")
    bb_exit = builder.append_basic_block(".exit")

    builder.cbranch(cmp_zero, bb_zero, bb_postest)

    with builder.goto_block(bb_zero):
        builder.store(ZERO, presult)
        builder.branch(bb_exit)

    with builder.goto_block(bb_postest):
        builder.cbranch(cmp_pos, bb_pos, bb_neg)

    with builder.goto_block(bb_pos):
        builder.store(POS, presult)
        builder.branch(bb_exit)

    with builder.goto_block(bb_neg):
        builder.store(NEG, presult)
        builder.branch(bb_exit)

    builder.position_at_end(bb_exit)
    res = builder.load(presult)
    return impl_ret_untracked(context, builder, sig.return_type, res)


builtin(implement('==', types.boolean, types.boolean)(int_eq_impl))
builtin(implement('!=', types.boolean, types.boolean)(int_ne_impl))
builtin(implement('<', types.boolean, types.boolean)(int_ult_impl))
builtin(implement('<=', types.boolean, types.boolean)(int_ule_impl))
builtin(implement('>', types.boolean, types.boolean)(int_ugt_impl))
builtin(implement('>=', types.boolean, types.boolean)(int_uge_impl))
builtin(implement('~', types.boolean)(bool_invert_impl))


def _implement_integer_operators():
    ty = types.Kind(types.Integer)

    builtin(implement('+', ty, ty)(int_add_impl))
    builtin(implement('-', ty, ty)(int_sub_impl))
    builtin(implement('*', ty, ty)(int_mul_impl))
    builtin(implement('==', ty, ty)(int_eq_impl))
    builtin(implement('!=', ty, ty)(int_ne_impl))

    builtin(implement('<<', ty, ty)(int_shl_impl))
    builtin(implement('>>', ty, ty)(int_shr_impl))

    builtin(implement('&', ty, ty)(int_and_impl))
    builtin(implement('|', ty, ty)(int_or_impl))
    builtin(implement('^', ty, ty)(int_xor_impl))

    builtin(implement('-', ty)(int_negate_impl))
    builtin(implement('+', ty)(int_positive_impl))
    builtin(implement(types.neg_type, ty)(int_negate_impl))
    builtin(implement('~', ty)(int_invert_impl))
    builtin(implement(types.sign_type, ty)(int_sign_impl))

    builtin(implement('**', ty, ty)(int_power_impl))
    builtin(implement(pow, ty, ty)(int_power_impl))

    for ty in types.unsigned_domain:
        builtin(implement('<', ty, ty)(int_ult_impl))
        builtin(implement('<=', ty, ty)(int_ule_impl))
        builtin(implement('>', ty, ty)(int_ugt_impl))
        builtin(implement('>=', ty, ty)(int_uge_impl))
        builtin(implement('**', types.float64, ty)(int_power_impl))
        builtin(implement(pow, types.float64, ty)(int_power_impl))
        builtin(implement(types.abs_type, ty)(uint_abs_impl))

    for ty in types.signed_domain:
        builtin(implement('<', ty, ty)(int_slt_impl))
        builtin(implement('<=', ty, ty)(int_sle_impl))
        builtin(implement('>', ty, ty)(int_sgt_impl))
        builtin(implement('>=', ty, ty)(int_sge_impl))
        builtin(implement(types.abs_type, ty)(int_abs_impl))
        builtin(implement('**', types.float64, ty)(int_power_impl))
        builtin(implement(pow, types.float64, ty)(int_power_impl))

_implement_integer_operators()


def optional_is_none(context, builder, sig, args):
    """Check if an Optional value is invalid
    """
    [lty, rty] = sig.args
    [lval, rval] = args

    # Make sure None is on the right
    if lty == types.none:
        lty, rty = rty, lty
        lval, rval = rval, lval

    opt_type = lty
    opt_val = lval

    del lty, rty, lval, rval

    opt = context.make_optional(opt_type)(context, builder, opt_val)
    res = builder.not_(cgutils.as_bool_bit(builder, opt.valid))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def optional_is_not_none(context, builder, sig, args):
    """Check if an Optional value is valid
    """
    res = builder.not_(optional_is_none(context, builder, sig, args))
    return impl_ret_untracked(context, builder, sig.return_type, res)


# None is/not None
builtin(implement('is', types.none, types.none)(
    optional.always_return_true_impl))

# Optional is None
builtin(implement('is',
                  types.Kind(types.Optional), types.none)(optional_is_none))

builtin(implement('is',
                  types.none, types.Kind(types.Optional))(optional_is_none))


def real_add_impl(context, builder, sig, args):
    res = builder.fadd(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_sub_impl(context, builder, sig, args):
    res = builder.fsub(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_mul_impl(context, builder, sig, args):
    res = builder.fmul(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_div_impl(context, builder, sig, args):
    with cgutils.if_zero(builder, args[1]):
        context.error_model.fp_zero_division(builder, ("division by zero",))
    res = builder.fdiv(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_divmod(context, builder, x, y):
    assert x.type == y.type
    floatty = x.type

    module = builder.module
    fname = ".numba.python.rem.%s" % x.type
    fnty = Type.function(floatty, (floatty, floatty, Type.pointer(floatty)))
    fn = module.get_or_insert_function(fnty, fname)

    if fn.is_declaration:
        fn.linkage = lc.LINKAGE_LINKONCE_ODR
        fnbuilder = lc.Builder.new(fn.append_basic_block('entry'))
        fx, fy, pmod = fn.args
        div, mod = real_divmod_func_body(context, fnbuilder, fx, fy)
        fnbuilder.store(mod, pmod)
        fnbuilder.ret(div)

    pmod = cgutils.alloca_once(builder, floatty)
    quotient = builder.call(fn, (x, y, pmod))
    return quotient, builder.load(pmod)


def real_divmod_func_body(context, builder, vx, wx):
    # Reference Objects/floatobject.c
    #
    # float_divmod(PyObject *v, PyObject *w)
    # {
    #     double vx, wx;
    #     double div, mod, floordiv;
    #     CONVERT_TO_DOUBLE(v, vx);
    #     CONVERT_TO_DOUBLE(w, wx);
    #     mod = fmod(vx, wx);
    #     /* fmod is typically exact, so vx-mod is *mathematically* an
    #        exact multiple of wx.  But this is fp arithmetic, and fp
    #        vx - mod is an approximation; the result is that div may
    #        not be an exact integral value after the division, although
    #        it will always be very close to one.
    #     */
    #     div = (vx - mod) / wx;
    #     if (mod) {
    #         /* ensure the remainder has the same sign as the denominator */
    #         if ((wx < 0) != (mod < 0)) {
    #             mod += wx;
    #             div -= 1.0;
    #         }
    #     }
    #     else {
    #         /* the remainder is zero, and in the presence of signed zeroes
    #            fmod returns different results across platforms; ensure
    #            it has the same sign as the denominator; we'd like to do
    #            "mod = wx * 0.0", but that may get optimized away */
    #         mod *= mod;  /* hide "mod = +0" from optimizer */
    #         if (wx < 0.0)
    #             mod = -mod;
    #     }
    #     /* snap quotient to nearest integral value */
    #     if (div) {
    #         floordiv = floor(div);
    #         if (div - floordiv > 0.5)
    #             floordiv += 1.0;
    #     }
    #     else {
    #         /* div is zero - get the same sign as the true quotient */
    #         div *= div;             /* hide "div = +0" from optimizers */
    #         floordiv = div * vx / wx; /* zero w/ sign of vx/wx */
    #     }
    #     return Py_BuildValue("(dd)", floordiv, mod);
    # }
    pmod = cgutils.alloca_once(builder, vx.type)
    pdiv = cgutils.alloca_once(builder, vx.type)
    pfloordiv = cgutils.alloca_once(builder, vx.type)

    mod = builder.frem(vx, wx)
    div = builder.fdiv(builder.fsub(vx, mod), wx)

    builder.store(mod, pmod)
    builder.store(div, pdiv)

    ZERO = Constant.real(vx.type, 0)
    ONE = Constant.real(vx.type, 1)
    mod_istrue = builder.fcmp(lc.FCMP_ONE, mod, ZERO)
    wx_ltz = builder.fcmp(lc.FCMP_OLT, wx, ZERO)
    mod_ltz = builder.fcmp(lc.FCMP_OLT, mod, ZERO)

    with builder.if_then(mod_istrue):
        wx_ltz_ne_mod_ltz = builder.icmp(lc.ICMP_NE, wx_ltz, mod_ltz)

        with builder.if_then(wx_ltz_ne_mod_ltz):
            mod = builder.fadd(mod, wx)
            div = builder.fsub(div, ONE)
            builder.store(mod, pmod)
            builder.store(div, pdiv)

    del mod
    del div

    with cgutils.ifnot(builder, mod_istrue):
        mod = builder.load(pmod)
        mod = builder.fmul(mod, mod)
        builder.store(mod, pmod)
        del mod

        with builder.if_then(wx_ltz):
            mod = builder.load(pmod)
            mod = builder.fsub(ZERO, mod)
            builder.store(mod, pmod)
            del mod

    div = builder.load(pdiv)
    div_istrue = builder.fcmp(lc.FCMP_ONE, div, ZERO)

    with builder.if_then(div_istrue):
        module = builder.module
        floorfn = lc.Function.intrinsic(module, lc.INTR_FLOOR, [wx.type])
        floordiv = builder.call(floorfn, [div])
        floordivdiff = builder.fsub(div, floordiv)
        floordivincr = builder.fadd(floordiv, ONE)
        HALF = Constant.real(wx.type, 0.5)
        pred = builder.fcmp(lc.FCMP_OGT, floordivdiff, HALF)
        floordiv = builder.select(pred, floordivincr, floordiv)
        builder.store(floordiv, pfloordiv)

    with cgutils.ifnot(builder, div_istrue):
        div = builder.fmul(div, div)
        builder.store(div, pdiv)
        floordiv = builder.fdiv(builder.fmul(div, vx), wx)
        builder.store(floordiv, pfloordiv)

    return builder.load(pfloordiv), builder.load(pmod)


def real_mod_impl(context, builder, sig, args):
    x, y = args
    res = cgutils.alloca_once(builder, x.type)
    with builder.if_else(cgutils.is_scalar_zero(builder, y), likely=False
                         ) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(
                builder, ("modulo by zero",)):
                # No exception raised => compute the nan result,
                # and set the FP exception word for Numpy warnings.
                rem = builder.frem(x, y)
                builder.store(rem, res)
        with if_non_zero:
            _, rem = real_divmod(context, builder, x, y)
            builder.store(rem, res)
    return impl_ret_untracked(context, builder, sig.return_type,
                              builder.load(res))


def real_floordiv_impl(context, builder, sig, args):
    x, y = args
    res = cgutils.alloca_once(builder, x.type)
    with builder.if_else(cgutils.is_scalar_zero(builder, y), likely=False
                         ) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(
                builder, ("division by zero",)):
                # No exception raised => compute the +/-inf or nan result,
                # and set the FP exception word for Numpy warnings.
                quot = builder.fdiv(x, y)
                builder.store(quot, res)
        with if_non_zero:
            quot, _ = real_divmod(context, builder, x, y)
            builder.store(quot, res)
    return impl_ret_untracked(context, builder, sig.return_type,
                              builder.load(res))


def real_power_impl(context, builder, sig, args):
    x, y = args
    module = builder.module
    if context.implement_powi_as_math_call:
        imp = context.get_function(math.pow, sig)
        res = imp(builder, args)
    else:
        fn = lc.Function.intrinsic(module, lc.INTR_POW, [y.type])
        res = builder.call(fn, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_lt_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_OLT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_le_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_OLE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_gt_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_OGT, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_ge_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_OGE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_eq_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_OEQ, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_ne_impl(context, builder, sig, args):
    res = builder.fcmp(lc.FCMP_UNE, *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_abs_impl(context, builder, sig, args):
    [ty] = sig.args
    sig = typing.signature(ty, ty)
    impl = context.get_function(math.fabs, sig)
    return impl(builder, args)


def real_negate_impl(context, builder, sig, args):
    from . import mathimpl
    res = mathimpl.negate_real(builder, args[0])
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_positive_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    res = context.cast(builder, val, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_sign_impl(context, builder, sig, args):
    [x] = args
    POS = Constant.real(x.type, 1)
    NEG = Constant.real(x.type, -1)
    ZERO = Constant.real(x.type, 0)

    presult = cgutils.alloca_once(builder, x.type)

    is_pos = builder.fcmp(lc.FCMP_OGT, x, ZERO)
    is_neg = builder.fcmp(lc.FCMP_OLT, x, ZERO)

    with builder.if_else(is_pos) as (gt_zero, not_gt_zero):
        with gt_zero:
            builder.store(POS, presult)
        with not_gt_zero:
            with builder.if_else(is_neg) as (lt_zero, not_lt_zero):
                with lt_zero:
                    builder.store(NEG, presult)
                with not_lt_zero:
                    # For both NaN and 0, the result of sign() is simply
                    # the input value.
                    builder.store(x, presult)

    res = builder.load(presult)
    return impl_ret_untracked(context, builder, sig.return_type, res)


ty = types.Kind(types.Float)

builtin(implement('+', ty, ty)(real_add_impl))
builtin(implement('-', ty, ty)(real_sub_impl))
builtin(implement('*', ty, ty)(real_mul_impl))
builtin(implement('/?', ty, ty)(real_div_impl))
builtin(implement('//', ty, ty)(real_floordiv_impl))
builtin(implement('/', ty, ty)(real_div_impl))
builtin(implement('%', ty, ty)(real_mod_impl))
builtin(implement('**', ty, ty)(real_power_impl))
builtin(implement(pow, ty, ty)(real_power_impl))

builtin(implement('==', ty, ty)(real_eq_impl))
builtin(implement('!=', ty, ty)(real_ne_impl))
builtin(implement('<', ty, ty)(real_lt_impl))
builtin(implement('<=', ty, ty)(real_le_impl))
builtin(implement('>', ty, ty)(real_gt_impl))
builtin(implement('>=', ty, ty)(real_ge_impl))

builtin(implement(types.abs_type, ty)(real_abs_impl))

builtin(implement('-', ty)(real_negate_impl))
builtin(implement('+', ty)(real_positive_impl))
builtin(implement(types.neg_type, ty)(real_negate_impl))
builtin(implement(types.sign_type, ty)(real_sign_impl))

del ty


class Complex64(cgutils.Structure):
    _fields = [('real', types.float32),
               ('imag', types.float32)]


class Complex128(cgutils.Structure):
    _fields = [('real', types.float64),
               ('imag', types.float64)]


def get_complex_info(ty):
    if ty == types.complex64:
        cmplxcls = Complex64
    elif ty == types.complex128:
        cmplxcls = Complex128
    else:
        raise TypeError(ty)

    return cmplxcls, ty.underlying_float


@builtin_attr
@impl_attribute(types.Kind(types.Complex), "real")
def complex_real_impl(context, builder, typ, value):
    cplx_cls = context.make_complex(typ)
    cplx = cplx_cls(context, builder, value=value)
    res = cplx.real
    return impl_ret_untracked(context, builder, typ, res)

@builtin_attr
@impl_attribute(types.Kind(types.Complex), "imag")
def complex_imag_impl(context, builder, typ, value):
    cplx_cls = context.make_complex(typ)
    cplx = cplx_cls(context, builder, value=value)
    res = cplx.imag
    return impl_ret_untracked(context, builder, typ, res)

@builtin
@implement("complex.conjugate", types.Kind(types.Complex))
def complex_conjugate_impl(context, builder, sig, args):
    from . import mathimpl
    cplx_cls = context.make_complex(sig.args[0])
    z = cplx_cls(context, builder, args[0])
    z.imag = mathimpl.negate_real(builder, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)

def real_real_impl(context, builder, typ, value):
    return impl_ret_untracked(context, builder, typ, value)

def real_imag_impl(context, builder, typ, value):
    res = cgutils.get_null_value(value.type)
    return impl_ret_untracked(context, builder, typ, res)

def real_conjugate_impl(context, builder, sig, args):
    return impl_ret_untracked(context, builder, sig.return_type, args[0])

for cls in (types.Float, types.Integer):
    builtin_attr(impl_attribute(types.Kind(cls), "real")(real_real_impl))
    builtin_attr(impl_attribute(types.Kind(cls), "imag")(real_imag_impl))
    builtin(implement("complex.conjugate", types.Kind(cls))(real_conjugate_impl))


@builtin
@implement("**", types.complex128, types.complex128)
@implement(pow, types.complex128, types.complex128)
def complex128_power_impl(context, builder, sig, args):
    [ca, cb] = args
    a = Complex128(context, builder, value=ca)
    b = Complex128(context, builder, value=cb)
    c = Complex128(context, builder)
    module = builder.module
    pa = a._getpointer()
    pb = b._getpointer()
    pc = c._getpointer()

    # Optimize for square because cpow looses a lot of precsiion
    TWO = context.get_constant(types.float64, 2)
    ZERO = context.get_constant(types.float64, 0)

    b_real_is_two = builder.fcmp(lc.FCMP_OEQ, b.real, TWO)
    b_imag_is_zero = builder.fcmp(lc.FCMP_OEQ, b.imag, ZERO)
    b_is_two = builder.and_(b_real_is_two, b_imag_is_zero)

    with builder.if_else(b_is_two) as (then, otherwise):
        with then:
            # Lower as multiplication
            res = complex_mul_impl(context, builder, sig, (ca, ca))
            cres = Complex128(context, builder, value=res)
            c.real = cres.real
            c.imag = cres.imag

        with otherwise:
            # Lower with call to external function
            fnty = Type.function(Type.void(), [pa.type] * 3)
            cpow = module.get_or_insert_function(fnty, name="numba.math.cpow")
            builder.call(cpow, (pa, pb, pc))

    res = builder.load(pc)
    return impl_ret_untracked(context, builder, sig.return_type, res)

def complex_add_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)
    z = complexClass(context, builder)
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    z.real = builder.fadd(a, c)
    z.imag = builder.fadd(b, d)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_sub_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)
    z = complexClass(context, builder)
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    z.real = builder.fsub(a, c)
    z.imag = builder.fsub(b, d)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_mul_impl(context, builder, sig, args):
    """
    (a+bi)(c+di)=(ac-bd)+i(ad+bc)
    """
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)
    z = complexClass(context, builder)
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    ac = builder.fmul(a, c)
    bd = builder.fmul(b, d)
    ad = builder.fmul(a, d)
    bc = builder.fmul(b, c)
    z.real = builder.fsub(ac, bd)
    z.imag = builder.fadd(ad, bc)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


NAN = float('nan')

def complex_div_impl(context, builder, sig, args):
    def complex_div(a, b):
        # This is CPython's algorithm (in _Py_c_quot()).
        areal = a.real
        aimag = a.imag
        breal = b.real
        bimag = b.imag
        if not breal and not bimag:
            raise ZeroDivisionError("complex division by zero")
        if abs(breal) >= abs(bimag):
            # Divide tops and bottom by b.real
            if not breal:
                return complex(NAN, NAN)
            ratio = bimag / breal
            denom = breal + bimag * ratio
            return complex(
                (areal + aimag * ratio) / denom,
                (aimag - areal * ratio) / denom)
        else:
            # Divide tops and bottom by b.imag
            if not bimag:
                return complex(NAN, NAN)
            ratio = breal / bimag
            denom = breal * ratio + bimag
            return complex(
                (a.real * ratio + a.imag) / denom,
                (a.imag * ratio - a.real) / denom)

    res = context.compile_internal(builder, complex_div, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_negate_impl(context, builder, sig, args):
    from . import mathimpl
    [typ] = sig.args
    [val] = args
    cmplxcls = context.make_complex(typ)
    cmplx = cmplxcls(context, builder, value=val)
    res = cmplxcls(context, builder)
    res.real = mathimpl.negate_real(builder, cmplx.real)
    res.imag = mathimpl.negate_real(builder, cmplx.imag)
    res = res._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_positive_impl(context, builder, sig, args):
    [val] = args
    return impl_ret_untracked(context, builder, sig.return_type, val)


def complex_eq_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)

    reals_are_eq = builder.fcmp(lc.FCMP_OEQ, x.real, y.real)
    imags_are_eq = builder.fcmp(lc.FCMP_OEQ, x.imag, y.imag)
    res = builder.and_(reals_are_eq, imags_are_eq)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_ne_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)

    reals_are_ne = builder.fcmp(lc.FCMP_UNE, x.real, y.real)
    imags_are_ne = builder.fcmp(lc.FCMP_UNE, x.imag, y.imag)
    res = builder.or_(reals_are_ne, imags_are_ne)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_abs_impl(context, builder, sig, args):
    """
    abs(z) := hypot(z.real, z.imag)
    """
    def complex_abs(z):
        return math.hypot(z.real, z.imag)

    res = context.compile_internal(builder, complex_abs, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


ty = types.Kind(types.Complex)

builtin(implement("+", ty, ty)(complex_add_impl))
builtin(implement("-", ty, ty)(complex_sub_impl))
builtin(implement("*", ty, ty)(complex_mul_impl))
builtin(implement("/?", ty, ty)(complex_div_impl))
builtin(implement("/", ty, ty)(complex_div_impl))
builtin(implement("-", ty)(complex_negate_impl))
builtin(implement("+", ty)(complex_positive_impl))
# Complex modulo is deprecated in python3

builtin(implement('==', ty, ty)(complex_eq_impl))
builtin(implement('!=', ty, ty)(complex_ne_impl))

builtin(implement(types.abs_type, ty)(complex_abs_impl))

del ty


#------------------------------------------------------------------------------


def number_not_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    istrue = context.cast(builder, val, typ, sig.return_type)
    res = builder.not_(istrue)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(bool, types.boolean)
def bool_as_bool(context, builder, sig, args):
    [val] = args
    return val

@builtin
@implement(bool, types.Kind(types.Integer))
def int_as_bool(context, builder, sig, args):
    [val] = args
    return builder.icmp_unsigned('!=', val, ir.Constant(val.type, 0))

@builtin
@implement(bool, types.Kind(types.Float))
def float_as_bool(context, builder, sig, args):
    [val] = args
    return builder.fcmp(lc.FCMP_UNE, val, ir.Constant(val.type, 0.0))

@builtin
@implement(bool, types.Kind(types.Complex))
def complex_as_bool(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    cmplx = context.make_complex(typ)(context, builder, val)
    real, imag = cmplx.real, cmplx.imag
    zero = ir.Constant(real.type, 0.0)
    real_istrue = builder.fcmp(lc.FCMP_UNE, real, zero)
    imag_istrue = builder.fcmp(lc.FCMP_UNE, imag, zero)
    return builder.or_(real_istrue, imag_istrue)


for ty in (types.Integer, types.Float, types.Complex):
    builtin(implement('not', types.Kind(ty))(number_not_impl))

builtin(implement('not', types.boolean)(number_not_impl))

#------------------------------------------------------------------------------

def make_pair(first_type, second_type):
    return cgutils.create_struct_proxy(types.Pair(first_type, second_type))


@builtin
@implement('getitem', types.Kind(types.CPointer), types.Kind(types.Integer))
def getitem_cpointer(context, builder, sig, args):
    base_ptr, idx = args
    elem_ptr = builder.gep(base_ptr, [idx])
    res = builder.load(elem_ptr)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement('setitem', types.Kind(types.CPointer), types.Kind(types.Integer),
           types.Any)
def setitem_cpointer(context, builder, sig, args):
    base_ptr, idx, val = args
    elem_ptr = builder.gep(base_ptr, [idx])
    builder.store(val, elem_ptr)


#-------------------------------------------------------------------------------

@builtin
@implement(types.NumberClass, types.Any)
def number_constructor(context, builder, sig, args):
    """
    Convert any number to any other.
    """
    [val] = args
    [valty] = sig.args
    return context.cast(builder, val, valty, sig.return_type)


#-------------------------------------------------------------------------------

@builtin
@implement(max, types.VarArg(types.Any))
def max_impl(context, builder, sig, args):
    argtys = sig.args
    for a in argtys:
        if a not in types.number_domain:
            raise AssertionError("only implemented for numeric types")

    def domax(a, b):
        at, av = a
        bt, bv = b
        ty = context.typing_context.unify_types(at, bt)
        cav = context.cast(builder, av, at, ty)
        cbv = context.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        ge = context.get_function(">=", cmpsig)
        pred = ge(builder, (cav, cbv))
        res = builder.select(pred, cav, cbv)
        return ty, res

    typvals = zip(argtys, args)
    resty, resval = reduce(domax, typvals)
    return impl_ret_borrowed(context, builder, sig.return_type, resval)


@builtin
@implement(min, types.VarArg(types.Any))
def min_impl(context, builder, sig, args):
    argtys = sig.args
    for a in argtys:
        if a not in types.number_domain:
            raise AssertionError("only implemented for numeric types")

    def domax(a, b):
        at, av = a
        bt, bv = b
        ty = context.typing_context.unify_types(at, bt)
        cav = context.cast(builder, av, at, ty)
        cbv = context.cast(builder, bv, bt, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        le = context.get_function("<=", cmpsig)
        pred = le(builder, (cav, cbv))
        res = builder.select(pred, cav, cbv)
        return ty, res

    typvals = zip(argtys, args)
    resty, resval = reduce(domax, typvals)
    return impl_ret_borrowed(context, builder, sig.return_type, resval)


def _round_intrinsic(tp):
    # round() rounds half to even on Python 3, away from zero on Python 2.
    if utils.IS_PY3:
        return "llvm.rint.f%d" % (tp.bitwidth,)
    else:
        return "llvm.round.f%d" % (tp.bitwidth,)

@builtin
@implement(round, types.Kind(types.Float))
def round_impl_unary(context, builder, sig, args):
    fltty = sig.args[0]
    llty = context.get_value_type(fltty)
    module = builder.module
    fnty = Type.function(llty, [llty])
    fn = module.get_or_insert_function(fnty, name=_round_intrinsic(fltty))
    res = builder.call(fn, args)
    if utils.IS_PY3:
        # unary round() returns an int on Python 3
        res = builder.fptosi(res, context.get_value_type(sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(round, types.Kind(types.Float), types.Kind(types.Integer))
def round_impl_binary(context, builder, sig, args):
    fltty = sig.args[0]
    # Allow calling the intrinsic from the Python implementation below.
    # This avoids the conversion to an int in Python 3's unary round().
    _round = types.ExternalFunction(
        _round_intrinsic(fltty), typing.signature(fltty, fltty))

    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x

        if ndigits >= 0:
            if ndigits > 22:
                # pow1 and pow2 are each safe from overflow, but
                # pow1*pow2 ~= pow(10.0, ndigits) might overflow.
                pow1 = 10.0 ** (ndigits - 22)
                pow2 = 1e22
            else:
                pow1 = 10.0 ** ndigits
                pow2 = 1.0
            y = (x * pow1) * pow2
            if math.isinf(y):
                return x
            return (_round(y) / pow2) / pow1

        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return _round(y) * pow1

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


#-------------------------------------------------------------------------------

@builtin
@implement(int, types.Any)
def int_impl(context, builder, sig, args):
    [ty] = sig.args
    [val] = args
    res = context.cast(builder, val, ty, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(float, types.Any)
def float_impl(context, builder, sig, args):
    [ty] = sig.args
    [val] = args
    res = context.cast(builder, val, ty, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(complex, types.VarArg(types.Any))
def complex_impl(context, builder, sig, args):
    complex_type = sig.return_type
    float_type = complex_type.underlying_float
    complex_cls = context.make_complex(complex_type)
    if len(sig.args) == 1:
        [argty] = sig.args
        [arg] = args
        if isinstance(argty, types.Complex):
            # Cast Complex* to Complex*
            res = context.cast(builder, arg, argty, complex_type)
            return impl_ret_untracked(context, builder, sig.return_type, res)
        else:
            real = context.cast(builder, arg, argty, float_type)
            imag = context.get_constant(float_type, 0)

    elif len(sig.args) == 2:
        [realty, imagty] = sig.args
        [real, imag] = args
        real = context.cast(builder, real, realty, float_type)
        imag = context.cast(builder, imag, imagty, float_type)

    cmplx = complex_cls(context, builder)
    cmplx.real = real
    cmplx.imag = imag
    res = cmplx._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)

# -----------------------------------------------------------------------------

@builtin_attr
@impl_attribute(types.Module(math), "pi", types.float64)
def math_pi_impl(context, builder, typ, value):
    res = context.get_constant(types.float64, math.pi)
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Module(math), "e", types.float64)
def math_e_impl(context, builder, typ, value):
    res = context.get_constant(types.float64, math.e)
    return impl_ret_untracked(context, builder, typ, res)

# -----------------------------------------------------------------------------

@builtin
@implement(intrinsics.array_ravel, types.Kind(types.Array))
def array_ravel_impl(context, builder, sig, args):
    [arrty] = sig.args
    [arr] = args
    flatarrty = sig.return_type

    flatarrcls = context.make_array(flatarrty)
    arrcls = context.make_array(arrty)

    flatarr = flatarrcls(context, builder)
    arr = arrcls(context, builder, value=arr)

    shapes = cgutils.unpack_tuple(builder, arr.shape, arrty.ndim)
    size = reduce(builder.mul, shapes)
    strides = cgutils.unpack_tuple(builder, arr.strides, arrty.ndim)
    unit_stride = strides[0] if arrty.layout == 'F' else strides[-1]

    context.populate_array(flatarr,
                           data=arr.data,
                           shape=cgutils.pack_array(builder, [size]),
                           strides=cgutils.pack_array(builder, [unit_stride]),
                           itemsize=arr.itemsize,
                           parent=arr.parent)

    res = flatarr._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------

@builtin
@implement(type, types.Any)
def type_impl(context, builder, sig, args):
    """
    One-argument type() builtin.
    """
    return context.get_dummy_value()
