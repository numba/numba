import math

from llvmlite import ir
from llvmlite.ir import Constant

from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, cgutils

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


def _get_power_zerodiv_return(context, return_type):
    if (isinstance(return_type, types.Integer)
        and not context.error_model.raise_on_fp_zero_division):
        # If not raising, return 0x8000... when computing 0 ** <negative number>
        return -1 << (return_type.bitwidth - 1)
    else:
        return False


def int_power_impl(context, builder, sig, args):
    """
    a ^ b, where a is an integer or real, and b an integer
    """
    is_integer = isinstance(sig.args[0], types.Integer)
    tp = sig.return_type
    zerodiv_return = _get_power_zerodiv_return(context, tp)

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
    res = builder.icmp_signed('<', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sle_impl(context, builder, sig, args):
    res = builder.icmp_signed('<=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sgt_impl(context, builder, sig, args):
    res = builder.icmp_signed('>', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sge_impl(context, builder, sig, args):
    res = builder.icmp_signed('>=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ult_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('<', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ule_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('<=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ugt_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('>', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_uge_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('>=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_eq_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('==', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_ne_impl(context, builder, sig, args):
    res = builder.icmp_unsigned('!=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_signed_unsigned_cmp(op):
    def impl(context, builder, sig, args):
        (left, right) = args
        # This code is translated from the NumPy source.
        # What we're going to do is divide the range of a signed value at zero.
        # If the signed value is less than zero, then we can treat zero as the
        # unsigned value since the unsigned value is necessarily zero or larger
        # and any signed comparison between a negative value and zero/infinity
        # will yield the same result. If the signed value is greater than or
        # equal to zero, then we can safely cast it to an unsigned value and do
        # the expected unsigned-unsigned comparison operation.
        # Original: https://github.com/numpy/numpy/pull/23713
        cmp_zero = builder.icmp_signed('<', left, Constant(left.type, 0))
        lt_zero = builder.icmp_signed(op, left, Constant(left.type, 0))
        ge_zero = builder.icmp_unsigned(op, left, right)
        res = builder.select(cmp_zero, lt_zero, ge_zero)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return impl


def int_unsigned_signed_cmp(op):
    def impl(context, builder, sig, args):
        (left, right) = args
        # See the function `int_signed_unsigned_cmp` for implementation notes.
        cmp_zero = builder.icmp_signed('<', right, Constant(right.type, 0))
        lt_zero = builder.icmp_signed(op, Constant(right.type, 0), right)
        ge_zero = builder.icmp_unsigned(op, left, right)
        res = builder.select(cmp_zero, lt_zero, ge_zero)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return impl


def int_abs_impl(context, builder, sig, args):
    [x] = args
    ZERO = Constant(x.type, None)
    ltz = builder.icmp_signed('<', x, ZERO)
    negated = builder.neg(x)
    res = builder.select(ltz, negated, x)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def identity_impl(context, builder, sig, args):
    [x] = args
    return impl_ret_untracked(context, builder, sig.return_type, x)


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
    res = builder.xor(val, Constant(val.type, int('1' * val.type.width, 2)))
    res = context.cast(builder, res, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def int_sign_impl(context, builder, sig, args):
    """
    np.sign(int)
    """
    [x] = args
    POS = Constant(x.type, 1)
    NEG = Constant(x.type, -1)
    ZERO = Constant(x.type, 0)

    cmp_zero = builder.icmp_unsigned('==', x, ZERO)
    cmp_pos = builder.icmp_signed('>', x, ZERO)

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


def real_add_impl(context, builder, sig, args):
    res = builder.fadd(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_sub_impl(context, builder, sig, args):
    res = builder.fsub(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_mul_impl(context, builder, sig, args):
    res = builder.fmul(*args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_power_impl(context, builder, sig, args):
    x, y = args
    module = builder.module
    if context.implement_powi_as_math_call:
        imp = context.get_function(math.pow, sig)
        res = imp(builder, args)
    else:
        fn = module.declare_intrinsic('llvm.pow', [y.type])
        res = builder.call(fn, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_lt_impl(context, builder, sig, args):
    res = builder.fcmp_ordered('<', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_le_impl(context, builder, sig, args):
    res = builder.fcmp_ordered('<=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_gt_impl(context, builder, sig, args):
    res = builder.fcmp_ordered('>', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_ge_impl(context, builder, sig, args):
    res = builder.fcmp_ordered('>=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_eq_impl(context, builder, sig, args):
    res = builder.fcmp_ordered('==', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_ne_impl(context, builder, sig, args):
    res = builder.fcmp_unordered('!=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_abs_impl(context, builder, sig, args):
    [ty] = sig.args
    sig = typing.signature(ty, ty)
    impl = context.get_function(math.fabs, sig)
    return impl(builder, args)


def real_negate_impl(context, builder, sig, args):
    from numba.cpython import mathimpl
    res = mathimpl.negate_real(builder, args[0])
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_positive_impl(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    res = context.cast(builder, val, typ, sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_sign_impl(context, builder, sig, args):
    """
    np.sign(float)
    """
    [x] = args
    POS = Constant(x.type, 1)
    NEG = Constant(x.type, -1)
    ZERO = Constant(x.type, 0)

    presult = cgutils.alloca_once(builder, x.type)

    is_pos = builder.fcmp_ordered('>', x, ZERO)
    is_neg = builder.fcmp_ordered('<', x, ZERO)

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


def complex_conjugate_impl(context, builder, sig, args):
    from numba.cpython import mathimpl
    z = context.make_complex(builder, sig.args[0], args[0])
    z.imag = mathimpl.negate_real(builder, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def real_conjugate_impl(context, builder, sig, args):
    return impl_ret_untracked(context, builder, sig.return_type, args[0])


def complex_power_impl(context, builder, sig, args):
    [ca, cb] = args
    ty = sig.args[0]
    fty = ty.underlying_float
    a = context.make_helper(builder, ty, value=ca)
    b = context.make_helper(builder, ty, value=cb)
    c = context.make_helper(builder, ty)
    module = builder.module
    pa = a._getpointer()
    pb = b._getpointer()
    pc = c._getpointer()

    # Optimize for square because cpow loses a lot of precision
    TWO = context.get_constant(fty, 2)
    ZERO = context.get_constant(fty, 0)

    b_real_is_two = builder.fcmp_ordered('==', b.real, TWO)
    b_imag_is_zero = builder.fcmp_ordered('==', b.imag, ZERO)
    b_is_two = builder.and_(b_real_is_two, b_imag_is_zero)

    with builder.if_else(b_is_two) as (then, otherwise):
        with then:
            # Lower as multiplication
            res = complex_mul_impl(context, builder, sig, (ca, ca))
            cres = context.make_helper(builder, ty, value=res)
            c.real = cres.real
            c.imag = cres.imag

        with otherwise:
            # Lower with call to external function
            func_name = {
                types.complex64: "numba_cpowf",
                types.complex128: "numba_cpow",
                }[ty]
            fnty = ir.FunctionType(ir.VoidType(), [pa.type] * 3)
            cpow = cgutils.get_or_insert_function(module, fnty, func_name)
            builder.call(cpow, (pa, pb, pc))

    res = builder.load(pc)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_add_impl(context, builder, sig, args):
    [cx, cy] = args
    ty = sig.args[0]
    x = context.make_complex(builder, ty, value=cx)
    y = context.make_complex(builder, ty, value=cy)
    z = context.make_complex(builder, ty)
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
    ty = sig.args[0]
    x = context.make_complex(builder, ty, value=cx)
    y = context.make_complex(builder, ty, value=cy)
    z = context.make_complex(builder, ty)
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
    ty = sig.args[0]
    x = context.make_complex(builder, ty, value=cx)
    y = context.make_complex(builder, ty, value=cy)
    z = context.make_complex(builder, ty)
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


def complex_negate_impl(context, builder, sig, args):
    from numba.cpython import mathimpl
    [typ] = sig.args
    [val] = args
    cmplx = context.make_complex(builder, typ, value=val)
    res = context.make_complex(builder, typ)
    res.real = mathimpl.negate_real(builder, cmplx.real)
    res.imag = mathimpl.negate_real(builder, cmplx.imag)
    res = res._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def complex_positive_impl(context, builder, sig, args):
    [val] = args
    return impl_ret_untracked(context, builder, sig.return_type, val)


def complex_abs_impl(context, builder, sig, args):
    """
    abs(z) := hypot(z.real, z.imag)
    """
    def complex_abs(z):
        return math.hypot(z.real, z.imag)

    res = context.compile_internal(builder, complex_abs, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)
