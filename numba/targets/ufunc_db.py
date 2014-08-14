"""This file contains information on how to translate different ufuncs
into numba. It is a database of different ufuncs and how each of its
loops maps to a function that implements the inner kernel of that ufunc
(the inner kernel being the per-element function).
"""

from __future__ import print_function, division, absolute_import


from .. import cgutils, types
from llvm import core as lc

def np_int_sdiv_impl(context, builder, sig, args):
    # based on the actual code in NumPy loops.c.src for signed integer types
    num, den = args
    lltype = num.type
    assert all(i.type==lltype for i in args), "must have homogeneous types"

    ZERO = lc.Constant.int(lltype, 0)
    MINUS_ONE = lc.Constant.int(lltype, -1)
    MIN_INT = lc.Constant.int(lltype, 1 << (den.type.width-1))
    den_is_zero = builder.icmp(lc.ICMP_EQ, ZERO, den)
    den_is_minus_one = builder.icmp(lc.ICMP_EQ, MINUS_ONE, den)
    num_is_min_int = builder.icmp(lc.ICMP_EQ, MIN_INT, num)
    could_cause_sigfpe = builder.and_(den_is_minus_one, num_is_min_int)
    force_zero = builder.or_(den_is_zero, could_cause_sigfpe)
    with cgutils.ifelse(builder, force_zero, expect=False) as (then, otherwise):
        with then:
            bb_then = builder.basic_block
        with otherwise:
            bb_otherwise = builder.basic_block
            div = builder.sdiv(num, den)
            mod = builder.srem(num, den)
            num_gt_zero = builder.icmp(lc.ICMP_SGT, num, ZERO)
            den_gt_zero = builder.icmp(lc.ICMP_SGT, den, ZERO)
            not_same_sign = builder.xor(num_gt_zero, den_gt_zero)
            mod_not_zero = builder.icmp(lc.ICMP_NE, mod, ZERO)
            needs_fixing = builder.and_(not_same_sign, mod_not_zero)
            fix_value = builder.select(needs_fixing, MINUS_ONE, ZERO)
            result_otherwise = builder.add(div, fix_value)

    result = builder.phi(lltype)
    result.add_incoming(ZERO, bb_then)
    result.add_incoming(result_otherwise, bb_otherwise)

    return result


def np_int_udiv_impl(context, builder, sig, args):
    num, den = args
    lltype = num.type
    assert all(i.type==lltype for i in args), "must have homogeneous types"

    ZERO = lc.Constant.int(lltype, 0)
    div_by_zero = builder.icmp(lc.ICMP_EQ, ZERO, den)
    with cgutils.ifelse(builder, div_by_zero, expect=False) as (then, otherwise):
        with then:
            # division by zero
            bb_then = builder.basic_block
        with otherwise:
            # divide!
            div = builder.udiv(num, den)
            bb_otherwise = builder.basic_block
    result = builder.phi(lltype)
    result.add_incoming(ZERO, bb_then)
    result.add_incoming(div, bb_otherwise)
    return result

def np_real_div_impl(context, builder, sig, args):
    return builder.fdiv(*args)

def _fabs(context, builder, arg):
    ZERO = lc.Constant.real(arg.type, 0.0)
    arg_negated = builder.fsub(ZERO, arg)
    arg_is_negative = builder.fcmp(lc.FCMP_OLT, arg, ZERO)
    return builder.select(arg_is_negative, arg_negated, arg)

def np_complex_div_impl(context, builder, sig, args):
    """extracted from numpy/core/src/umath/loops.c.src,
    inspired by complex_div_impl"""

    complexClass = context.make_complex(sig.args[0])
    in1, in2 = [complexClass(context, builder, value=arg) for arg in args]

    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type==ftype for i in [in1r, in1i, in2r, in2i]]), "mismatched types"
    presult_real = cgutils.alloca_once(builder, ftype)
    presult_imag = cgutils.alloca_once(builder, ftype)

    ZERO = lc.Constant.real(ftype, 0.0)
    ONE = lc.Constant.real(ftype, 1.0)
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)

    in2r_abs_ge_in2i_abs = builder.fcmp(lc.FCMP_OGE, in2r_abs, in2i_abs)

    with cgutils.ifelse(builder, in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            in2r_is_zero = builder.fcmp(lc.FCMP_OEQ, in2r_abs, ZERO)
            in2i_is_zero = builder.fcmp(lc.FCMP_OEQ, in2i_abs, ZERO)
            in2_is_zero = builder.and_(in2r_is_zero, in2i_is_zero)
            with cgutils.ifelse(builder, in2_is_zero) as (inn_then, inn_otherwise):
                with inn_then:
                    real = builder.fdiv(in1r, in2r_abs)
                    imag = builder.fdiv(in1i, in2i_abs)
                    builder.store(real, presult_real)
                    builder.store(imag, presult_imag)
                with inn_otherwise:
                    rat = builder.fdiv(in2i, in2r)
                    # sc1 = 1.0/(in2r + in2i*rat)
                    tmp1 = builder.fmul(in2i, rat)
                    tmp2 = builder.fadd(in2r, tmp1)
                    scl = builder.fdiv(ONE, tmp2)
                    # out.real = (in1r + in1i*rat)*scl
                    # out.imag = (in1i - in1r*rat)*scl
                    tmp3 = builder.fmul(in1i, rat)
                    tmp4 = builder.fmul(in1r, rat)
                    tmp5 = builder.fadd(in1r, tmp3)
                    tmp6 = builder.fsub(in1i, tmp4)
                    real = builder.fmul(tmp5, scl)
                    imag = builder.fmul(tmp6, scl)
                    builder.store(real, presult_real)
                    builder.store(imag, presult_imag)
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            # scl = 1.0/(in2i + in2r*rat)
            tmp1 = builder.fmul(in2r, rat)
            tmp2 = builder.fadd(in2i, tmp1)
            scl = builder.fdiv(ONE, tmp2)
            # out.real = (in1r*rat + in1i)*scl
            # out.imag = (in1i*rat - in1r)*scl
            tmp3 = builder.fmul(in1r, rat)
            tmp4 = builder.fmul(in1i, rat)
            tmp5 = builder.fadd(tmp3, in1i)
            tmp6 = builder.fsub(tmp4, in1r)
            real = builder.fmul(tmp5, scl)
            imag = builder.fmul(tmp6, scl)
            builder.store(real, presult_real)
            builder.store(imag, presult_imag)

    out = complexClass(context, builder)
    out.real = builder.load(presult_real)
    out.imag = builder.load(presult_imag)
    return out._getvalue()


class UFuncDB(object):
    def __init__(self):
        self.dict = {}

    def __getitem__(self, key):
        try:
            return self.dict[key]
        except KeyError:
            if not self.dict:
                _fill_ufunc_db(self.dict)
                return self.dict[key]
            else:
                raise

ufunc_db = UFuncDB()

def _fill_ufunc_db(ufunc_db):
    from . import builtins
    import numpy as np

    ufunc_db[np.negative] = {
        '?->?': builtins.number_not_impl,
        'b->b': builtins.int_negate_impl,
        'B->B': builtins.int_negate_impl,
        'h->h': builtins.int_negate_impl,
        'H->H': builtins.int_negate_impl,
        'i->i': builtins.int_negate_impl,
        'I->I': builtins.int_negate_impl,
        'l->l': builtins.int_negate_impl,
        'L->L': builtins.int_negate_impl,
        'q->q': builtins.int_negate_impl,
        'Q->Q': builtins.int_negate_impl,
        'f->f': builtins.real_negate_impl,
        'd->d': builtins.real_negate_impl,
        'F->F': builtins.complex_negate_impl,
        'D->D': builtins.complex_negate_impl,
    }

    ufunc_db[np.absolute] = {
        '?->?': builtins.int_abs_impl,
        'b->b': builtins.int_abs_impl,
        'B->B': builtins.uint_abs_impl,
        'h->h': builtins.int_abs_impl,
        'H->H': builtins.uint_abs_impl,
        'i->i': builtins.int_abs_impl,
        'I->I': builtins.uint_abs_impl,
        'l->l': builtins.int_abs_impl,
        'L->L': builtins.uint_abs_impl,
        'q->q': builtins.int_abs_impl,
        'Q->Q': builtins.uint_abs_impl,
        'f->f': builtins.real_abs_impl,
        'd->d': builtins.real_abs_impl,
    }

    ufunc_db[np.add] = {
        '??->?': builtins.int_or_impl,
        'bb->b': builtins.int_add_impl,
        'BB->B': builtins.int_add_impl,
        'hh->h': builtins.int_add_impl,
        'HH->H': builtins.int_add_impl,
        'ii->i': builtins.int_add_impl,
        'II->I': builtins.int_add_impl,
        'll->l': builtins.int_add_impl,
        'LL->L': builtins.int_add_impl,
        'qq->q': builtins.int_add_impl,
        'QQ->Q': builtins.int_add_impl,
        'ff->f': builtins.real_add_impl,
        'dd->d': builtins.real_add_impl,
        'FF->F': builtins.complex_add_impl,
        'DD->D': builtins.complex_add_impl,
    }

    ufunc_db[np.subtract] = {
        '??->?': builtins.int_xor_impl,
        'bb->b': builtins.int_sub_impl,
        'BB->B': builtins.int_sub_impl,
        'hh->h': builtins.int_sub_impl,
        'HH->H': builtins.int_sub_impl,
        'ii->i': builtins.int_sub_impl,
        'II->I': builtins.int_sub_impl,
        'll->l': builtins.int_sub_impl,
        'LL->L': builtins.int_sub_impl,
        'qq->q': builtins.int_sub_impl,
        'QQ->Q': builtins.int_sub_impl,
        'ff->f': builtins.real_sub_impl,
        'dd->d': builtins.real_sub_impl,
        'FF->F': builtins.complex_sub_impl,
        'DD->D': builtins.complex_sub_impl,
    }

    ufunc_db[np.multiply] = {
        '??->?': builtins.int_and_impl,
        'bb->b': builtins.int_mul_impl,
        'BB->B': builtins.int_mul_impl,
        'hh->h': builtins.int_mul_impl,
        'HH->H': builtins.int_mul_impl,
        'ii->i': builtins.int_mul_impl,
        'II->I': builtins.int_mul_impl,
        'll->l': builtins.int_mul_impl,
        'LL->L': builtins.int_mul_impl,
        'qq->q': builtins.int_mul_impl,
        'QQ->Q': builtins.int_mul_impl,
        'ff->f': builtins.real_mul_impl,
        'dd->d': builtins.real_mul_impl,
        'FF->F': builtins.complex_mul_impl,
        'DD->D': builtins.complex_mul_impl,
    }

    if np.divide != np.true_divide:
        ufunc_db[np.divide] = {
            'bb->b': np_int_sdiv_impl,
            'BB->B': np_int_udiv_impl,
            'hh->h': np_int_sdiv_impl,
            'HH->H': np_int_udiv_impl,
            'ii->i': np_int_sdiv_impl,
            'II->I': np_int_udiv_impl,
            'll->l': np_int_sdiv_impl,
            'LL->L': np_int_udiv_impl,
            'qq->q': np_int_sdiv_impl,
            'QQ->Q': np_int_udiv_impl,
            'ff->f': np_real_div_impl,
            'dd->d': np_real_div_impl,
            'FF->F': np_complex_div_impl,
            'DD->D': np_complex_div_impl,
        }
