from __future__ import print_function, absolute_import, division
import math
from functools import reduce

from llvm.core import Type, Constant
import llvm.core as lc

from numba import errcode
from numba import types, typing, cgutils, utils
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iterator_impl, iternext_impl,
                                    struct_factory)
from numba.typing import signature


#-------------------------------------------------------------------------------


def int_add_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    return builder.add(a, b)


def int_sub_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    return builder.sub(a, b)


def int_mul_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    return builder.mul(a, b)


def int_udiv_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    cgutils.guard_zero(context, builder, b)
    return builder.udiv(a, b)


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
    with cgutils.ifthen(builder, cond):
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


def int_sdiv_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    cgutils.guard_zero(context, builder, b)
    div, _ = int_divmod(context, builder, a, b)
    return div


def int_struediv_impl(context, builder, sig, args):
    x, y = args
    fx = builder.sitofp(x, Type.double())
    fy = builder.sitofp(y, Type.double())
    cgutils.guard_zero(context, builder, y)
    return builder.fdiv(fx, fy)


def int_utruediv_impl(context, builder, sig, args):
    x, y = args
    fx = builder.uitofp(x, Type.double())
    fy = builder.uitofp(y, Type.double())
    cgutils.guard_zero(context, builder, y)
    return builder.fdiv(fx, fy)


int_sfloordiv_impl = int_sdiv_impl
int_ufloordiv_impl = int_udiv_impl


def int_srem_impl(context, builder, sig, args):
    x, y = args
    cgutils.guard_zero(context, builder, y)
    _, rem = int_divmod(context, builder, x, y)
    return rem


def int_urem_impl(context, builder, sig, args):
    x, y = args
    return builder.urem(x, y)


def int_spower_impl(context, builder, sig, args):
    module = cgutils.get_module(builder)
    x, y = args
    if y.type.width > 32:
        y = builder.trunc(y, Type.int(32))
    elif y.type.width < 32:
        y = builder.sext(y, Type.int(32))

    if context.implement_powi_as_math_call:
        undersig = typing.signature(sig.return_type, sig.args[0], types.int32)
        impl = context.get_function(math.pow, undersig)
        return impl(builder, (x, y))
    else:
        powerfn = lc.Function.intrinsic(module, lc.INTR_POWI, [x.type])
        return builder.call(powerfn, (x, y))


def int_upower_impl(context, builder, sig, args):
    module = cgutils.get_module(builder)
    x, y = args
    if y.type.width > 32:
        y = builder.trunc(y, Type.int(32))
    elif y.type.width < 32:
        y = builder.zext(y, Type.int(32))

    if context.implement_powi_as_math_call:
        undersig = typing.signature(sig.return_type, sig.args[0], types.int32)
        impl = context.get_function(math.pow, undersig)
        return impl(builder, (x, y))
    else:
        powerfn = lc.Function.intrinsic(module, lc.INTR_POWI, [x.type])
        return builder.call(powerfn, (x, y))


def int_power_func_body(context, builder, x, y):
    pcounter = cgutils.alloca_once(builder, y.type)
    presult = cgutils.alloca_once(builder, x.type)
    result = Constant.int(x.type, 1)
    counter = y
    builder.store(counter, pcounter)
    builder.store(result, presult)

    bbcond = cgutils.append_basic_block(builder, ".cond")
    bbbody = cgutils.append_basic_block(builder, ".body")
    bbexit = cgutils.append_basic_block(builder, ".exit")

    del counter
    del result

    builder.branch(bbcond)

    with cgutils.goto_block(builder, bbcond):
        counter = builder.load(pcounter)
        ONE = Constant.int(counter.type, 1)
        ZERO = Constant.null(counter.type)
        builder.store(builder.sub(counter, ONE), pcounter)
        pred = builder.icmp(lc.ICMP_SGT, counter, ZERO)
        builder.cbranch(pred, bbbody, bbexit)

    with cgutils.goto_block(builder, bbbody):
        result = builder.load(presult)
        builder.store(builder.mul(result, x), presult)
        builder.branch(bbcond)

    builder.position_at_end(bbexit)
    return builder.load(presult)


def int_slt_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_SLT, *args)


def int_sle_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_SLE, *args)


def int_sgt_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_SGT, *args)


def int_sge_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_SGE, *args)


def int_ult_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_ULT, *args)


def int_ule_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_ULE, *args)


def int_ugt_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_UGT, *args)


def int_uge_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_UGE, *args)


def int_eq_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_EQ, *args)


def int_ne_impl(context, builder, sig, args):
    return builder.icmp(lc.ICMP_NE, *args)


def int_abs_impl(context, builder, sig, args):
    [x] = args
    ZERO = Constant.null(x.type)
    ltz = builder.icmp(lc.ICMP_SLT, x, ZERO)
    negated = builder.neg(x)
    return builder.select(ltz, negated, x)


def uint_abs_impl(context, builder, sig, args):
    [x] = args
    return x


def int_shl_impl(context, builder, sig, args):
    [valty, amtty] = sig.args
    [val, amt] = args
    val = context.cast(builder, val, valty, sig.return_type)
    amt = context.cast(builder, amt, amtty, sig.return_type)
    return builder.shl(val, amt)


def int_lshr_impl(context, builder, sig, args):
    [valty, amtty] = sig.args
    [val, amt] = args
    val = context.cast(builder, val, valty, sig.return_type)
    amt = context.cast(builder, amt, amtty, sig.return_type)
    return builder.lshr(val, amt)


def int_ashr_impl(context, builder, sig, args):
    [valty, amtty] = sig.args
    [val, amt] = args
    val = context.cast(builder, val, valty, sig.return_type)
    amt = context.cast(builder, amt, amtty, sig.return_type)
    return builder.ashr(val, amt)


def int_and_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    return builder.and_(cav, cbc)


def int_or_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    return builder.or_(cav, cbc)


def int_xor_impl(context, builder, sig, args):
    [at, bt] = sig.args
    [av, bv] = args
    cav = context.cast(builder, av, at, sig.return_type)
    cbc = context.cast(builder, bv, bt, sig.return_type)
    return builder.xor(cav, cbc)


def int_negate_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    val = context.cast(builder, val, typ, sig.return_type)
    if sig.return_type in types.real_domain:
        return builder.fsub(context.get_constant(sig.return_type, 0), val)
    else:
        return builder.neg(val)


def int_positive_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    return context.cast(builder, val, typ, sig.return_type)


def int_invert_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    val = context.cast(builder, val, typ, sig.return_type)
    return builder.xor(val, Constant.all_ones(val.type))


def int_sign_impl(context, builder, sig, args):
    [x] = args
    POS = Constant.int(x.type, 1)
    NEG = Constant.int(x.type, -1)
    ZERO = Constant.int(x.type, 0)

    cmp_zero = builder.icmp(lc.ICMP_EQ, x, ZERO)
    cmp_pos = builder.icmp(lc.ICMP_SGT, x, ZERO)

    presult = cgutils.alloca_once(builder, x.type)

    bb_zero = cgutils.append_basic_block(builder, ".zero")
    bb_postest = cgutils.append_basic_block(builder, ".postest")
    bb_pos = cgutils.append_basic_block(builder, ".pos")
    bb_neg = cgutils.append_basic_block(builder, ".neg")
    bb_exit = cgutils.append_basic_block(builder, ".exit")

    builder.cbranch(cmp_zero, bb_zero, bb_postest)

    with cgutils.goto_block(builder, bb_zero):
        builder.store(ZERO, presult)
        builder.branch(bb_exit)

    with cgutils.goto_block(builder, bb_postest):
        builder.cbranch(cmp_pos, bb_pos, bb_neg)

    with cgutils.goto_block(builder, bb_pos):
        builder.store(POS, presult)
        builder.branch(bb_exit)

    with cgutils.goto_block(builder, bb_neg):
        builder.store(NEG, presult)
        builder.branch(bb_exit)

    builder.position_at_end(bb_exit)
    return builder.load(presult)


builtin(implement('==', types.boolean, types.boolean)(int_eq_impl))
builtin(implement('!=', types.boolean, types.boolean)(int_ne_impl))
builtin(implement('<', types.boolean, types.boolean)(int_ult_impl))
builtin(implement('<=', types.boolean, types.boolean)(int_ule_impl))
builtin(implement('>', types.boolean, types.boolean)(int_ugt_impl))
builtin(implement('>=', types.boolean, types.boolean)(int_uge_impl))
builtin(implement('~', types.boolean)(int_invert_impl))

for ty in types.integer_domain:
    builtin(implement('+', ty, ty)(int_add_impl))
    builtin(implement('-', ty, ty)(int_sub_impl))
    builtin(implement('*', ty, ty)(int_mul_impl))
    builtin(implement('==', ty, ty)(int_eq_impl))
    builtin(implement('!=', ty, ty)(int_ne_impl))

    builtin(implement('<<', ty, types.uint32)(int_shl_impl))

    builtin(implement('&', ty, ty)(int_and_impl))
    builtin(implement('|', ty, ty)(int_or_impl))
    builtin(implement('^', ty, ty)(int_xor_impl))

    builtin(implement('-', ty)(int_negate_impl))
    builtin(implement('+', ty)(int_positive_impl))
    builtin(implement(types.neg_type, ty)(int_negate_impl))
    builtin(implement('~', ty)(int_invert_impl))
    builtin(implement(types.sign_type, ty)(int_sign_impl))

for ty in types.unsigned_domain:
    builtin(implement('/?', ty, ty)(int_udiv_impl))
    builtin(implement('//', ty, ty)(int_ufloordiv_impl))
    builtin(implement('/', ty, ty)(int_utruediv_impl))
    builtin(implement('%', ty, ty)(int_urem_impl))
    builtin(implement('<', ty, ty)(int_ult_impl))
    builtin(implement('<=', ty, ty)(int_ule_impl))
    builtin(implement('>', ty, ty)(int_ugt_impl))
    builtin(implement('>=', ty, ty)(int_uge_impl))
    builtin(implement('**', types.float64, ty)(int_upower_impl))
    # logical shift for unsigned
    builtin(implement('>>', ty, types.uint32)(int_lshr_impl))
    builtin(implement(types.abs_type, ty)(uint_abs_impl))

for ty in types.signed_domain:
    builtin(implement('/?', ty, ty)(int_sdiv_impl))
    builtin(implement('//', ty, ty)(int_sfloordiv_impl))
    builtin(implement('/', ty, ty)(int_struediv_impl))
    builtin(implement('%', ty, ty)(int_srem_impl))
    builtin(implement('<', ty, ty)(int_slt_impl))
    builtin(implement('<=', ty, ty)(int_sle_impl))
    builtin(implement('>', ty, ty)(int_sgt_impl))
    builtin(implement('>=', ty, ty)(int_sge_impl))
    builtin(implement(types.abs_type, ty)(int_abs_impl))
    builtin(implement('**', types.float64, ty)(int_spower_impl))
    # arithmetic shift for signed
    builtin(implement('>>', ty, types.uint32)(int_ashr_impl))


def real_add_impl(context, builder, sig, args):
    return builder.fadd(*args)


def real_sub_impl(context, builder, sig, args):
    return builder.fsub(*args)


def real_mul_impl(context, builder, sig, args):
    return builder.fmul(*args)


def real_div_impl(context, builder, sig, args):
    cgutils.guard_zero(context, builder, args[1])
    return builder.fdiv(*args)


def real_divmod(context, builder, x, y):
    assert x.type == y.type
    floatty = x.type

    module = cgutils.get_module(builder)
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

    with cgutils.ifthen(builder, mod_istrue):
        wx_ltz_ne_mod_ltz = builder.icmp(lc.ICMP_NE, wx_ltz, mod_ltz)

        with cgutils.ifthen(builder, wx_ltz_ne_mod_ltz):
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

        with cgutils.ifthen(builder, wx_ltz):
            mod = builder.load(pmod)
            mod = builder.fsub(ZERO, mod)
            builder.store(mod, pmod)
            del mod

    div = builder.load(pdiv)
    div_istrue = builder.fcmp(lc.FCMP_ONE, div, ZERO)

    with cgutils.ifthen(builder, div_istrue):
        module = cgutils.get_module(builder)
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
    cgutils.guard_zero(context, builder, y)
    _, rem = real_divmod(context, builder, x, y)
    return rem


def real_floordiv_impl(context, builder, sig, args):
    x, y = args
    cgutils.guard_zero(context, builder, y)
    quot, _ = real_divmod(context, builder, x, y)
    return quot


def real_power_impl(context, builder, sig, args):
    x, y = args
    module = cgutils.get_module(builder)
    if context.implement_powi_as_math_call:
        imp = context.get_function(math.pow, sig)
        return imp(builder, args)
    else:
        fn = lc.Function.intrinsic(module, lc.INTR_POW, [y.type])
        return builder.call(fn, (x, y))


def real_lt_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_OLT, *args)


def real_le_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_OLE, *args)


def real_gt_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_OGT, *args)


def real_ge_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_OGE, *args)


def real_eq_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_OEQ, *args)


def real_ne_impl(context, builder, sig, args):
    return builder.fcmp(lc.FCMP_UNE, *args)


def real_abs_impl(context, builder, sig, args):
    [ty] = sig.args
    sig = typing.signature(ty, ty)
    impl = context.get_function(math.fabs, sig)
    return impl(builder, args)

def real_negate_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    val = context.cast(builder, val, typ, sig.return_type)
    if sig.return_type in types.real_domain:
        return builder.fsub(context.get_constant(sig.return_type, 0), val)
    else:
        return builder.neg(val)


def real_positive_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    return context.cast(builder, val, typ, sig.return_type)


def real_sign_impl(context, builder, sig, args):
    [x] = args
    POS = Constant.real(x.type, 1)
    NEG = Constant.real(x.type, -1)
    ZERO = Constant.real(x.type, 0)

    cmp_zero = builder.fcmp(lc.FCMP_OEQ, x, ZERO)
    cmp_pos = builder.fcmp(lc.FCMP_OGT, x, ZERO)

    presult = cgutils.alloca_once(builder, x.type)

    bb_zero = cgutils.append_basic_block(builder, ".zero")
    bb_postest = cgutils.append_basic_block(builder, ".postest")
    bb_pos = cgutils.append_basic_block(builder, ".pos")
    bb_neg = cgutils.append_basic_block(builder, ".neg")
    bb_exit = cgutils.append_basic_block(builder, ".exit")

    builder.cbranch(cmp_zero, bb_zero, bb_postest)

    with cgutils.goto_block(builder, bb_zero):
        builder.store(ZERO, presult)
        builder.branch(bb_exit)

    with cgutils.goto_block(builder, bb_postest):
        builder.cbranch(cmp_pos, bb_pos, bb_neg)

    with cgutils.goto_block(builder, bb_pos):
        builder.store(POS, presult)
        builder.branch(bb_exit)

    with cgutils.goto_block(builder, bb_neg):
        builder.store(NEG, presult)
        builder.branch(bb_exit)

    builder.position_at_end(bb_exit)
    return builder.load(presult)


for ty in types.real_domain:
    builtin(implement('+', ty, ty)(real_add_impl))
    builtin(implement('-', ty, ty)(real_sub_impl))
    builtin(implement('*', ty, ty)(real_mul_impl))
    builtin(implement('/?', ty, ty)(real_div_impl))
    builtin(implement('//', ty, ty)(real_floordiv_impl))
    builtin(implement('/', ty, ty)(real_div_impl))
    builtin(implement('%', ty, ty)(real_mod_impl))
    builtin(implement('**', ty, ty)(real_power_impl))

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
@impl_attribute(types.complex64, "real", types.float32)
def complex64_real_impl(context, builder, typ, value):
    cplx = Complex64(context, builder, value=value)
    return cplx.real


@builtin_attr
@impl_attribute(types.complex128, "real", types.float64)
def complex128_real_impl(context, builder, typ, value):
    cplx = Complex128(context, builder, value=value)
    return cplx.real


@builtin_attr
@impl_attribute(types.complex64, "imag", types.float32)
def complex64_imag_impl(context, builder, typ, value):
    cplx = Complex64(context, builder, value=value)
    return cplx.imag


@builtin_attr
@impl_attribute(types.complex128, "imag", types.float64)
def complex128_imag_impl(context, builder, typ, value):
    cplx = Complex128(context, builder, value=value)
    return cplx.imag


@builtin
@implement("**", types.complex128, types.complex128)
def complex128_power_impl(context, builder, sig, args):
    [ca, cb] = args
    a = Complex128(context, builder, value=ca)
    b = Complex128(context, builder, value=cb)
    c = Complex128(context, builder)
    module = cgutils.get_module(builder)
    pa = a._getpointer()
    pb = b._getpointer()
    pc = c._getpointer()

    # Optimize for square because cpow looses a lot of precsiion
    TWO = context.get_constant(types.float64, 2)
    ZERO = context.get_constant(types.float64, 0)

    b_real_is_two = builder.fcmp(lc.FCMP_OEQ, b.real, TWO)
    b_imag_is_zero = builder.fcmp(lc.FCMP_OEQ, b.imag, ZERO)
    b_is_two = builder.and_(b_real_is_two, b_imag_is_zero)

    with cgutils.ifelse(builder, b_is_two) as (then, otherwise):
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

    return builder.load(pc)


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
    return z._getvalue()


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
    return z._getvalue()


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
    return z._getvalue()


def complex_div_impl(context, builder, sig, args):
    """
    z = c^2 + d^2
    (a+bi)/(c+di) = (ac + bd) / z, (bc - ad) / z
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

    cc = builder.fmul(c, c)
    dd = builder.fmul(d, d)
    zz = builder.fadd(cc, dd)

    ac_bd = builder.fadd(ac, bd)
    bc_ad = builder.fsub(bc, ad)

    cgutils.guard_zero(context, builder, zz)
    z.real = builder.fdiv(ac_bd, zz)
    z.imag = builder.fdiv(bc_ad, zz)
    return z._getvalue()


def complex_negate_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    cmplxcls = context.make_complex(typ)
    cmplx = cmplxcls(context, builder, val)

    real = cmplx.real
    imag = cmplx.imag

    zero = Constant.real(real.type, 0)

    res = cmplxcls(context, builder)
    res.real = builder.fsub(zero, real)
    res.imag = builder.fsub(zero, imag)
    return res._getvalue()


def complex_positive_impl(context, builder, sig, args):
    [val] = args
    return val


def complex_eq_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)

    reals_are_eq = builder.fcmp(lc.FCMP_OEQ, x.real, y.real)
    imags_are_eq = builder.fcmp(lc.FCMP_OEQ, x.imag, y.imag)
    return builder.and_(reals_are_eq, imags_are_eq)


def complex_ne_impl(context, builder, sig, args):
    [cx, cy] = args
    complexClass = context.make_complex(sig.args[0])
    x = complexClass(context, builder, value=cx)
    y = complexClass(context, builder, value=cy)

    reals_are_ne = builder.fcmp(lc.FCMP_UNE, x.real, y.real)
    imags_are_ne = builder.fcmp(lc.FCMP_UNE, x.imag, y.imag)
    return builder.or_(reals_are_ne, imags_are_ne)


def complex_abs_impl(context, builder, sig, args):
    """
    abs(z) := hypot(z.real, z.imag)
    """
    [typ] = sig.args
    [val] = args
    cmplxcls = context.make_complex(typ)
    flty = typ.underlying_float
    cmplx = cmplxcls(context, builder, val)
    [x, y] = cmplx.real, cmplx.imag
    hypotsig = signature(sig.return_type, flty, flty)
    hypotimp = context.get_function(math.hypot, hypotsig)
    return hypotimp(builder, [x, y])


for ty, cls in zip([types.complex64, types.complex128],
                   [Complex64, Complex128]):
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


#------------------------------------------------------------------------------


def number_not_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    istrue = context.cast(builder, val, typ, sig.return_type)
    return builder.not_(istrue)


def number_as_bool_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    istrue = context.cast(builder, val, typ, sig.return_type)
    return istrue


for ty in types.number_domain:
    builtin(implement('not', ty)(number_not_impl))
    builtin(implement(bool, ty)(number_as_bool_impl))

builtin(implement('not', types.boolean)(number_not_impl))

#------------------------------------------------------------------------------

class Slice(cgutils.Structure):
    _fields = [('start', types.intp),
               ('stop', types.intp),
               ('step', types.intp), ]


@builtin
@implement(types.slice_type, types.intp, types.intp, types.intp)
def slice3_impl(context, builder, sig, args):
    start, stop, step = args

    slice3 = Slice(context, builder)
    slice3.start = start
    slice3.stop = stop
    slice3.step = step

    return slice3._getvalue()


@builtin
@implement(types.slice_type, types.intp, types.intp)
def slice2_impl(context, builder, sig, args):
    start, stop = args

    slice3 = Slice(context, builder)
    slice3.start = start
    slice3.stop = stop
    slice3.step = context.get_constant(types.intp, 1)

    return slice3._getvalue()


@builtin
@implement(types.slice_type, types.intp, types.none)
def slice1_start_impl(context, builder, sig, args):
    start, stop = args

    slice3 = Slice(context, builder)
    slice3.start = start
    maxint = (1 << (context.address_size - 1)) - 1
    slice3.stop = context.get_constant(types.intp, maxint)
    slice3.step = context.get_constant(types.intp, 1)

    return slice3._getvalue()


@builtin
@implement(types.slice_type, types.none, types.intp)
def slice1_stop_impl(context, builder, sig, args):
    start, stop = args

    slice3 = Slice(context, builder)
    slice3.start = context.get_constant(types.intp, 0)
    slice3.stop = stop
    slice3.step = context.get_constant(types.intp, 1)

    return slice3._getvalue()


@builtin
@implement(types.slice_type)
def slice0_empty_impl(context, builder, sig, args):
    assert not args

    slice3 = Slice(context, builder)
    slice3.start = context.get_constant(types.intp, 0)
    maxint = (1 << (context.address_size - 1)) - 1
    slice3.stop = context.get_constant(types.intp, maxint)
    slice3.step = context.get_constant(types.intp, 1)

    return slice3._getvalue()


@builtin
@implement(types.slice_type, types.none, types.none)
def slice0_none_none_impl(context, builder, sig, args):
    assert len(args) == 2
    newsig = typing.signature(types.slice_type)
    return slice0_empty_impl(context, builder, newsig, ())


def make_pair(first_type, second_type):
    class Pair(cgutils.Structure):
        _fields = [('first', first_type),
                   ('second', second_type)]

    return Pair


@struct_factory(types.UniTupleIter)
def make_unituple_iter(tupiter):
    """
    Return the Structure representation of the given *tupiter* (an
    instance of types.UniTupleIter).
    """
    unituple = tupiter.unituple

    class UniTupleIter(cgutils.Structure):
        _fields = [('index', types.CPointer(types.intp)),
                   ('tuple', tupiter.unituple,)]

    return UniTupleIter


@builtin
@implement('getiter', types.Kind(types.UniTuple))
def getiter_unituple(context, builder, sig, args):
    [tupty] = sig.args
    [tup] = args

    tupitercls = make_unituple_iter(types.UniTupleIter(tupty))
    iterval = tupitercls(context, builder)

    index0 = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once(builder, index0.type)
    builder.store(index0, indexptr)

    iterval.index = indexptr
    iterval.tuple = tup

    return iterval._getvalue()


# Unfortunately, we can't make decorate UniTupleIter with iterator_impl
# as it would register the iternext() method too late. It really has to
# be registered at module import time, not while compiling.

@builtin
@implement('iternext', types.Kind(types.UniTupleIter))
@iternext_impl
def iternext_unituple(context, builder, sig, args, result):
    [tupiterty] = sig.args
    [tupiter] = args

    tupitercls = make_unituple_iter(tupiterty)
    iterval = tupitercls(context, builder, value=tupiter)
    tup = iterval.tuple
    idxptr = iterval.index
    idx = builder.load(idxptr)
    count = context.get_constant(types.intp, tupiterty.unituple.count)

    is_valid = builder.icmp(lc.ICMP_SLT, idx, count)
    result.set_valid(is_valid)

    with cgutils.ifthen(builder, is_valid):
        getitem_sig = typing.signature(sig.return_type, tupiterty.unituple,
                                       types.intp)
        result.yield_(getitem_unituple(context, builder, getitem_sig, [tup, idx]))
        nidx = builder.add(idx, context.get_constant(types.intp, 1))
        builder.store(nidx, iterval.index)


@builtin
@implement('getitem', types.Kind(types.UniTuple), types.intp)
def getitem_unituple(context, builder, sig, args):
    tupty, _ = sig.args
    tup, idx = args

    bbelse = cgutils.append_basic_block(builder, "switch.else")
    bbend = cgutils.append_basic_block(builder, "switch.end")
    switch = builder.switch(idx, bbelse, n=tupty.count)

    with cgutils.goto_block(builder, bbelse):
        context.return_errcode(builder, errcode.OUT_OF_BOUND_ERROR)

    lrtty = context.get_value_type(tupty.dtype)
    with cgutils.goto_block(builder, bbend):
        phinode = builder.phi(lrtty)

    for i in range(tupty.count):
        ki = context.get_constant(types.intp, i)
        bbi = cgutils.append_basic_block(builder, "switch.%d" % i)
        switch.add_case(ki, bbi)
        with cgutils.goto_block(builder, bbi):
            value = builder.extract_value(tup, i)
            builder.branch(bbend)
            phinode.add_incoming(value, bbi)

    builder.position_at_end(bbend)
    return phinode

#-------------------------------------------------------------------------------


def caster(restype):
    @implement(restype, types.Any)
    def _cast(context, builder, sig, args):
        [val] = args
        [valty] = sig.args
        return context.cast(builder, val, valty, restype)

    return _cast


builtin(caster(types.int8))
builtin(caster(types.int16))
builtin(caster(types.int32))
builtin(caster(types.int64))

builtin(caster(types.uint8))
builtin(caster(types.uint16))
builtin(caster(types.uint32))
builtin(caster(types.uint64))

builtin(caster(types.float32))
builtin(caster(types.float64))

builtin(caster(types.complex64))
builtin(caster(types.complex128))


#-------------------------------------------------------------------------------

@builtin
@implement(max, types.VarArg)
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
    return resval


@builtin
@implement(min, types.VarArg)
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
    return resval


@builtin
@implement(round, types.float32)
def round_impl_f32(context, builder, sig, args):
    module = cgutils.get_module(builder)
    fnty = Type.function(Type.float(), [Type.float()])
    if utils.IS_PY3:
        fn = module.get_or_insert_function(fnty, name="numba.roundf")
    else:
        fn = module.get_or_insert_function(fnty, name="roundf")
    assert fn.is_declaration
    return builder.call(fn, args)


@builtin
@implement(round, types.float64)
def round_impl_f64(context, builder, sig, args):
    module = cgutils.get_module(builder)
    fnty = Type.function(Type.double(), [Type.double()])
    if utils.IS_PY3:
        fn = module.get_or_insert_function(fnty, name="numba.round")
    else:
        fn = module.get_or_insert_function(fnty, name="round")
    assert fn.is_declaration
    return builder.call(fn, args)

#-------------------------------------------------------------------------------

@builtin
@implement(int, types.Any)
def int_impl(context, builder, sig, args):
    [ty] = sig.args
    [val] = args
    return context.cast(builder, val, ty, sig.return_type)


@builtin
@implement(float, types.Any)
def float_impl(context, builder, sig, args):
    [ty] = sig.args
    [val] = args
    return context.cast(builder, val, ty, sig.return_type)


@builtin
@implement(complex, types.VarArg)
def complex_impl(context, builder, sig, args):
    if len(sig.args) == 1:
        [argty] = sig.args
        [arg] = args
        if isinstance(argty, types.Complex):
            # Cast Complex* to Complex128
            return context.cast(builder, arg, argty, types.complex128)
        else:
            real = context.cast(builder, arg, argty, types.float64)
            imag = context.get_constant(types.float64, 0)

    elif len(sig.args) == 2:
        [realty, imagty] = sig.args
        [real, imag] = args
        real = context.cast(builder, real, realty, types.float64)
        imag = context.cast(builder, imag, imagty, types.float64)

    cmplx = Complex128(context, builder)
    cmplx.real = real
    cmplx.imag = imag
    return cmplx._getvalue()

# -----------------------------------------------------------------------------

@builtin_attr
@impl_attribute(types.Module(math), "pi", types.float64)
def math_pi_impl(context, builder, typ, value):
    return context.get_constant(types.float64, math.pi)


@builtin_attr
@impl_attribute(types.Module(math), "e", types.float64)
def math_e_impl(context, builder, typ, value):
    return context.get_constant(types.float64, math.e)

