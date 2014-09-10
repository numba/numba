"""This file contains information on how to translate different ufuncs
into numba. It is a database of different ufuncs and how each of its
loops maps to a function that implements the inner kernel of that ufunc
(the inner kernel being the per-element function).

Use the function get_ufunc_info to get the information related to the
ufunc
"""

from __future__ import print_function, division, absolute_import

import numpy as np


# this is lazily initialized to avoid circular imports
_ufunc_db = None

def _lazy_init_db():
    global _ufunc_db

    if _ufunc_db is None:
        _ufunc_db = {}
        _fill_ufunc_db(_ufunc_db)


def get_ufuncs():
    """obtain a list of supported ufuncs in the db"""
    _lazy_init_db()
    return _ufunc_db.keys()


def get_ufunc_info(ufunc_key):
    """get the lowering information for the ufunc with key ufunc_key.

    The lowering information is a dictionary that maps from a numpy
    loop string (as given by the ufunc types attribute) to a function
    that handles code generation for a scalar version of the ufunc
    (that is, generates the "per element" operation").

    raises a KeyError if the ufunc is not in the ufunc_db
    """
    _lazy_init_db()
    return _ufunc_db[ufunc_key]


def _fill_ufunc_db(ufunc_db):
    # some of these imports would cause a problem of circular
    # imports if done at global scope when importing the numba
    # module.
    from . import builtins, npyfuncs

    ufunc_db[np.negative] = {
        '?->?': builtins.bool_invert_impl,
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
        'F->f': builtins.complex_abs_impl,
        'D->d': builtins.complex_abs_impl,
    }

    ufunc_db[np.sign] = {
        'b->b': builtins.int_sign_impl,
        'B->B': builtins.int_sign_impl,
        'h->h': builtins.int_sign_impl,
        'H->H': builtins.int_sign_impl,
        'i->i': builtins.int_sign_impl,
        'I->I': builtins.int_sign_impl,
        'l->l': builtins.int_sign_impl,
        'L->L': builtins.int_sign_impl,
        'q->q': builtins.int_sign_impl,
        'Q->Q': builtins.int_sign_impl,
        'f->f': builtins.real_sign_impl,
        'd->d': builtins.real_sign_impl,
        'F->F': npyfuncs.np_complex_sign_impl,
        'D->D': npyfuncs.np_complex_sign_impl,
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
            'bb->b': npyfuncs.np_int_sdiv_impl,
            'BB->B': npyfuncs.np_int_udiv_impl,
            'hh->h': npyfuncs.np_int_sdiv_impl,
            'HH->H': npyfuncs.np_int_udiv_impl,
            'ii->i': npyfuncs.np_int_sdiv_impl,
            'II->I': npyfuncs.np_int_udiv_impl,
            'll->l': npyfuncs.np_int_sdiv_impl,
            'LL->L': npyfuncs.np_int_udiv_impl,
            'qq->q': npyfuncs.np_int_sdiv_impl,
            'QQ->Q': npyfuncs.np_int_udiv_impl,
            'ff->f': npyfuncs.np_real_div_impl,
            'dd->d': npyfuncs.np_real_div_impl,
            'FF->F': npyfuncs.np_complex_div_impl,
            'DD->D': npyfuncs.np_complex_div_impl,
        }

    ufunc_db[np.true_divide] = {
        'bb->d': npyfuncs.np_int_truediv_impl,
        'BB->d': npyfuncs.np_int_truediv_impl,
        'hh->d': npyfuncs.np_int_truediv_impl,
        'HH->d': npyfuncs.np_int_truediv_impl,
        'ii->d': npyfuncs.np_int_truediv_impl,
        'II->d': npyfuncs.np_int_truediv_impl,
        'll->d': npyfuncs.np_int_truediv_impl,
        'LL->d': npyfuncs.np_int_truediv_impl,
        'qq->d': npyfuncs.np_int_truediv_impl,
        'QQ->d': npyfuncs.np_int_truediv_impl,
        'ff->f': npyfuncs.np_real_div_impl,
        'dd->d': npyfuncs.np_real_div_impl,
        'FF->F': npyfuncs.np_complex_div_impl,
        'DD->D': npyfuncs.np_complex_div_impl,
    }

    ufunc_db[np.floor_divide] = {
        'bb->b': npyfuncs.np_int_sdiv_impl,
        'BB->B': npyfuncs.np_int_udiv_impl,
        'hh->h': npyfuncs.np_int_sdiv_impl,
        'HH->H': npyfuncs.np_int_udiv_impl,
        'ii->i': npyfuncs.np_int_sdiv_impl,
        'II->I': npyfuncs.np_int_udiv_impl,
        'll->l': npyfuncs.np_int_sdiv_impl,
        'LL->L': npyfuncs.np_int_udiv_impl,
        'qq->q': npyfuncs.np_int_sdiv_impl,
        'QQ->Q': npyfuncs.np_int_udiv_impl,
        'ff->f': npyfuncs.np_real_floor_div_impl,
        'dd->d': npyfuncs.np_real_floor_div_impl,
        'FF->F': npyfuncs.np_complex_floor_div_impl,
        'DD->D': npyfuncs.np_complex_floor_div_impl,
    }

    ufunc_db[np.remainder] = {
        'bb->b': npyfuncs.np_int_srem_impl,
        'BB->B': npyfuncs.np_int_urem_impl,
        'hh->h': npyfuncs.np_int_srem_impl,
        'HH->H': npyfuncs.np_int_urem_impl,
        'ii->i': npyfuncs.np_int_srem_impl,
        'II->I': npyfuncs.np_int_urem_impl,
        'll->l': npyfuncs.np_int_srem_impl,
        'LL->L': npyfuncs.np_int_urem_impl,
        'qq->q': npyfuncs.np_int_srem_impl,
        'QQ->Q': npyfuncs.np_int_urem_impl,
        'ff->f': npyfuncs.np_real_mod_impl,
        'dd->d': npyfuncs.np_real_mod_impl,
    }

    ufunc_db[np.fmod] = {
        'bb->b': npyfuncs.np_int_fmod_impl,
        'BB->B': npyfuncs.np_int_fmod_impl,
        'hh->h': npyfuncs.np_int_fmod_impl,
        'HH->H': npyfuncs.np_int_fmod_impl,
        'ii->i': npyfuncs.np_int_fmod_impl,
        'II->I': npyfuncs.np_int_fmod_impl,
        'll->l': npyfuncs.np_int_fmod_impl,
        'LL->L': npyfuncs.np_int_fmod_impl,
        'qq->q': npyfuncs.np_int_fmod_impl,
        'QQ->Q': npyfuncs.np_int_fmod_impl,
        'ff->f': npyfuncs.np_real_fmod_impl,
        'dd->d': npyfuncs.np_real_fmod_impl,
    }

    ufunc_db[np.logaddexp] = {
        'ff->f': npyfuncs.np_real_logaddexp_impl,
        'dd->d': npyfuncs.np_real_logaddexp_impl,
    }

    ufunc_db[np.logaddexp2] = {
        'ff->f': npyfuncs.np_real_logaddexp2_impl,
        'dd->d': npyfuncs.np_real_logaddexp2_impl,
    }

    ufunc_db[np.power] = {
        'bb->b': npyfuncs.np_int_power_impl,
        'BB->B': npyfuncs.np_int_power_impl,
        'hh->h': npyfuncs.np_int_power_impl,
        'HH->H': npyfuncs.np_int_power_impl,
        'ii->i': npyfuncs.np_int_power_impl,
        'II->I': npyfuncs.np_int_power_impl,
        'll->l': npyfuncs.np_int_power_impl,
        'LL->L': npyfuncs.np_int_power_impl,
        'qq->q': npyfuncs.np_int_power_impl,
        'QQ->Q': npyfuncs.np_int_power_impl,
        'ff->f': npyfuncs.np_real_power_impl,
        'dd->d': npyfuncs.np_real_power_impl,
        'FF->F': npyfuncs.np_complex_power_impl,
        'DD->D': npyfuncs.np_complex_power_impl,
    }

    ufunc_db[np.rint] = {
        'f->f': npyfuncs.np_real_rint_impl,
        'd->d': npyfuncs.np_real_rint_impl,
        'F->F': npyfuncs.np_complex_rint_impl,
        'D->D': npyfuncs.np_complex_rint_impl,
    }

    ufunc_db[np.conjugate] = {
        'b->b': npyfuncs.np_dummy_return_arg,
        'B->B': npyfuncs.np_dummy_return_arg,
        'h->h': npyfuncs.np_dummy_return_arg,
        'H->H': npyfuncs.np_dummy_return_arg,
        'i->i': npyfuncs.np_dummy_return_arg,
        'I->I': npyfuncs.np_dummy_return_arg,
        'l->l': npyfuncs.np_dummy_return_arg,
        'L->L': npyfuncs.np_dummy_return_arg,
        'q->q': npyfuncs.np_dummy_return_arg,
        'Q->Q': npyfuncs.np_dummy_return_arg,
        'f->f': npyfuncs.np_dummy_return_arg,
        'd->d': npyfuncs.np_dummy_return_arg,
        'F->F': npyfuncs.np_complex_conjugate_impl,
        'D->D': npyfuncs.np_complex_conjugate_impl,
    }

    ufunc_db[np.exp] = {
        'f->f': npyfuncs.np_real_exp_impl,
        'd->d': npyfuncs.np_real_exp_impl,
        'F->F': npyfuncs.np_complex_exp_impl,
        'D->D': npyfuncs.np_complex_exp_impl,
    }

    ufunc_db[np.exp2] = {
        'f->f': npyfuncs.np_real_exp2_impl,
        'd->d': npyfuncs.np_real_exp2_impl,
        'F->F': npyfuncs.np_complex_exp2_impl,
        'D->D': npyfuncs.np_complex_exp2_impl,
    }

    ufunc_db[np.log] = {
        'f->f': npyfuncs.np_real_log_impl,
        'd->d': npyfuncs.np_real_log_impl,
        'F->F': npyfuncs.np_complex_log_impl,
        'D->D': npyfuncs.np_complex_log_impl,
    }

    ufunc_db[np.log2] = {
        'f->f': npyfuncs.np_real_log2_impl,
        'd->d': npyfuncs.np_real_log2_impl,
        'F->F': npyfuncs.np_complex_log2_impl,
        'D->D': npyfuncs.np_complex_log2_impl,
    }

    ufunc_db[np.log10] = {
        'f->f': npyfuncs.np_real_log10_impl,
        'd->d': npyfuncs.np_real_log10_impl,
        'F->F': npyfuncs.np_complex_log10_impl,
        'D->D': npyfuncs.np_complex_log10_impl,
    }

    ufunc_db[np.expm1] = {
        'f->f': npyfuncs.np_real_expm1_impl,
        'd->d': npyfuncs.np_real_expm1_impl,
        'F->F': npyfuncs.np_complex_expm1_impl,
        'D->D': npyfuncs.np_complex_expm1_impl,
    }

    ufunc_db[np.log1p] = {
        'f->f': npyfuncs.np_real_log1p_impl,
        'd->d': npyfuncs.np_real_log1p_impl,
        'F->F': npyfuncs.np_complex_log1p_impl,
        'D->D': npyfuncs.np_complex_log1p_impl,
    }

    ufunc_db[np.sqrt] = {
        'f->f': npyfuncs.np_real_sqrt_impl,
        'd->d': npyfuncs.np_real_sqrt_impl,
        'F->F': npyfuncs.np_complex_sqrt_impl,
        'D->D': npyfuncs.np_complex_sqrt_impl,
    }

    ufunc_db[np.square] = {
        'b->b': npyfuncs.np_int_square_impl,
        'B->B': npyfuncs.np_int_square_impl,
        'h->h': npyfuncs.np_int_square_impl,
        'H->H': npyfuncs.np_int_square_impl,
        'i->i': npyfuncs.np_int_square_impl,
        'I->I': npyfuncs.np_int_square_impl,
        'l->l': npyfuncs.np_int_square_impl,
        'L->L': npyfuncs.np_int_square_impl,
        'q->q': npyfuncs.np_int_square_impl,
        'Q->Q': npyfuncs.np_int_square_impl,
        'f->f': npyfuncs.np_real_square_impl,
        'd->d': npyfuncs.np_real_square_impl,
        'F->F': npyfuncs.np_complex_square_impl,
        'D->D': npyfuncs.np_complex_square_impl,
    }

    ufunc_db[np.reciprocal] = {
        'b->b': npyfuncs.np_int_reciprocal_impl,
        'B->B': npyfuncs.np_int_reciprocal_impl,
        'h->h': npyfuncs.np_int_reciprocal_impl,
        'H->H': npyfuncs.np_int_reciprocal_impl,
        'i->i': npyfuncs.np_int_reciprocal_impl,
        'I->I': npyfuncs.np_int_reciprocal_impl,
        'l->l': npyfuncs.np_int_reciprocal_impl,
        'L->L': npyfuncs.np_int_reciprocal_impl,
        'q->q': npyfuncs.np_int_reciprocal_impl,
        'Q->Q': npyfuncs.np_int_reciprocal_impl,
        'f->f': npyfuncs.np_real_reciprocal_impl,
        'd->d': npyfuncs.np_real_reciprocal_impl,
        'F->F': npyfuncs.np_complex_reciprocal_impl,
        'D->D': npyfuncs.np_complex_reciprocal_impl,
    }

    ufunc_db[np.sin] = {
        'f->f': npyfuncs.np_real_sin_impl,
        'd->d': npyfuncs.np_real_sin_impl,
        'F->F': npyfuncs.np_complex_sin_impl,
        'D->D': npyfuncs.np_complex_sin_impl,
    }

    ufunc_db[np.cos] = {
        'f->f': npyfuncs.np_real_cos_impl,
        'd->d': npyfuncs.np_real_cos_impl,
        'F->F': npyfuncs.np_complex_cos_impl,
        'D->D': npyfuncs.np_complex_cos_impl,
    }

    ufunc_db[np.tan] = {
        'f->f': npyfuncs.np_real_tan_impl,
        'd->d': npyfuncs.np_real_tan_impl,
        'F->F': npyfuncs.np_complex_tan_impl,
        'D->D': npyfuncs.np_complex_tan_impl,
    }

    ufunc_db[np.arcsin] = {
        'f->f': npyfuncs.np_real_asin_impl,
        'd->d': npyfuncs.np_real_asin_impl,
        'F->F': npyfuncs.np_complex_asin_impl,
        'D->D': npyfuncs.np_complex_asin_impl,
    }

    ufunc_db[np.arccos] = {
        'f->f': npyfuncs.np_real_acos_impl,
        'd->d': npyfuncs.np_real_acos_impl,
        'F->F': npyfuncs.np_complex_acos_impl,
        'D->D': npyfuncs.np_complex_acos_impl,
    }

    ufunc_db[np.arctan] = {
        'f->f': npyfuncs.np_real_atan_impl,
        'd->d': npyfuncs.np_real_atan_impl,
        'F->F': npyfuncs.np_complex_atan_impl,
        'D->D': npyfuncs.np_complex_atan_impl,
    }

    ufunc_db[np.arctan2] = {
        'ff->f': npyfuncs.np_real_atan2_impl,
        'dd->d': npyfuncs.np_real_atan2_impl,
    }

    ufunc_db[np.hypot] = {
        'ff->f': npyfuncs.np_real_hypot_impl,
        'dd->d': npyfuncs.np_real_hypot_impl,
    }

    ufunc_db[np.sinh] = {
        'f->f': npyfuncs.np_real_sinh_impl,
        'd->d': npyfuncs.np_real_sinh_impl,
        'F->F': npyfuncs.np_complex_sinh_impl,
        'D->D': npyfuncs.np_complex_sinh_impl,
    }

    ufunc_db[np.cosh] = {
        'f->f': npyfuncs.np_real_cosh_impl,
        'd->d': npyfuncs.np_real_cosh_impl,
        'F->F': npyfuncs.np_complex_cosh_impl,
        'D->D': npyfuncs.np_complex_cosh_impl,
    }

    ufunc_db[np.tanh] = {
        'f->f': npyfuncs.np_real_tanh_impl,
        'd->d': npyfuncs.np_real_tanh_impl,
        'F->F': npyfuncs.np_complex_tanh_impl,
        'D->D': npyfuncs.np_complex_tanh_impl,
    }

    ufunc_db[np.arcsinh] = {
        'f->f': npyfuncs.np_real_asinh_impl,
        'd->d': npyfuncs.np_real_asinh_impl,
        'F->F': npyfuncs.np_complex_asinh_impl,
        'D->D': npyfuncs.np_complex_asinh_impl,
    }

    ufunc_db[np.arccosh] = {
        'f->f': npyfuncs.np_real_acosh_impl,
        'd->d': npyfuncs.np_real_acosh_impl,
        'F->F': npyfuncs.np_complex_acosh_impl,
        'D->D': npyfuncs.np_complex_acosh_impl,
    }

    ufunc_db[np.arctanh] = {
        'f->f': npyfuncs.np_real_atanh_impl,
        'd->d': npyfuncs.np_real_atanh_impl,
        'F->F': npyfuncs.np_complex_atanh_impl,
        'D->D': npyfuncs.np_complex_atanh_impl,
    }

    ufunc_db[np.deg2rad] = {
        'f->f': npyfuncs.np_real_deg2rad_impl,
        'd->d': npyfuncs.np_real_deg2rad_impl,
    }

    ufunc_db[np.radians] = ufunc_db[np.deg2rad]

    ufunc_db[np.rad2deg] = {
        'f->f': npyfuncs.np_real_rad2deg_impl,
        'd->d': npyfuncs.np_real_rad2deg_impl,
    }

    ufunc_db[np.degrees] = ufunc_db[np.rad2deg]

    ufunc_db[np.floor] = {
        'f->f': npyfuncs.np_real_floor_impl,
        'd->d': npyfuncs.np_real_floor_impl,
    }

    ufunc_db[np.ceil] = {
        'f->f': npyfuncs.np_real_ceil_impl,
        'd->d': npyfuncs.np_real_ceil_impl,
    }

    ufunc_db[np.trunc] = {
        'f->f': npyfuncs.np_real_trunc_impl,
        'd->d': npyfuncs.np_real_trunc_impl,
    }

    ufunc_db[np.fabs] = {
        'f->f': npyfuncs.np_real_fabs_impl,
        'd->d': npyfuncs.np_real_fabs_impl,
    }

    # logical ufuncs
    ufunc_db[np.greater] = {
        '??->?': builtins.int_ugt_impl,
        'bb->?': builtins.int_sgt_impl,
        'BB->?': builtins.int_ugt_impl,
        'hh->?': builtins.int_sgt_impl,
        'HH->?': builtins.int_ugt_impl,
        'ii->?': builtins.int_sgt_impl,
        'II->?': builtins.int_ugt_impl,
        'll->?': builtins.int_sgt_impl,
        'LL->?': builtins.int_ugt_impl,
        'qq->?': builtins.int_sgt_impl,
        'QQ->?': builtins.int_ugt_impl,
        'ff->?': builtins.real_gt_impl,
        'dd->?': builtins.real_gt_impl,
        'FF->?': npyfuncs.np_complex_gt_impl,
        'DD->?': npyfuncs.np_complex_gt_impl,
    }

    ufunc_db[np.greater_equal] = {
        '??->?': builtins.int_uge_impl,
        'bb->?': builtins.int_sge_impl,
        'BB->?': builtins.int_uge_impl,
        'hh->?': builtins.int_sge_impl,
        'HH->?': builtins.int_uge_impl,
        'ii->?': builtins.int_sge_impl,
        'II->?': builtins.int_uge_impl,
        'll->?': builtins.int_sge_impl,
        'LL->?': builtins.int_uge_impl,
        'qq->?': builtins.int_sge_impl,
        'QQ->?': builtins.int_uge_impl,
        'ff->?': builtins.real_ge_impl,
        'dd->?': builtins.real_ge_impl,
        'FF->?': npyfuncs.np_complex_ge_impl,
        'DD->?': npyfuncs.np_complex_ge_impl,
    }

    ufunc_db[np.less] = {
        '??->?': builtins.int_ult_impl,
        'bb->?': builtins.int_slt_impl,
        'BB->?': builtins.int_ult_impl,
        'hh->?': builtins.int_slt_impl,
        'HH->?': builtins.int_ult_impl,
        'ii->?': builtins.int_slt_impl,
        'II->?': builtins.int_ult_impl,
        'll->?': builtins.int_slt_impl,
        'LL->?': builtins.int_ult_impl,
        'qq->?': builtins.int_slt_impl,
        'QQ->?': builtins.int_ult_impl,
        'ff->?': builtins.real_lt_impl,
        'dd->?': builtins.real_lt_impl,
        'FF->?': npyfuncs.np_complex_lt_impl,
        'DD->?': npyfuncs.np_complex_lt_impl,
    }

    ufunc_db[np.less_equal] = {
        '??->?': builtins.int_ule_impl,
        'bb->?': builtins.int_sle_impl,
        'BB->?': builtins.int_ule_impl,
        'hh->?': builtins.int_sle_impl,
        'HH->?': builtins.int_ule_impl,
        'ii->?': builtins.int_sle_impl,
        'II->?': builtins.int_ule_impl,
        'll->?': builtins.int_sle_impl,
        'LL->?': builtins.int_ule_impl,
        'qq->?': builtins.int_sle_impl,
        'QQ->?': builtins.int_ule_impl,
        'ff->?': builtins.real_le_impl,
        'dd->?': builtins.real_le_impl,
        'FF->?': npyfuncs.np_complex_le_impl,
        'DD->?': npyfuncs.np_complex_le_impl,
    }

    ufunc_db[np.not_equal] = {
        '??->?': builtins.int_ne_impl,
        'bb->?': builtins.int_ne_impl,
        'BB->?': builtins.int_ne_impl,
        'hh->?': builtins.int_ne_impl,
        'HH->?': builtins.int_ne_impl,
        'ii->?': builtins.int_ne_impl,
        'II->?': builtins.int_ne_impl,
        'll->?': builtins.int_ne_impl,
        'LL->?': builtins.int_ne_impl,
        'qq->?': builtins.int_ne_impl,
        'QQ->?': builtins.int_ne_impl,
        'ff->?': builtins.real_ne_impl,
        'dd->?': builtins.real_ne_impl,
        'FF->?': npyfuncs.np_complex_ne_impl,
        'DD->?': npyfuncs.np_complex_ne_impl,
    }

    ufunc_db[np.equal] = {
        '??->?': builtins.int_eq_impl,
        'bb->?': builtins.int_eq_impl,
        'BB->?': builtins.int_eq_impl,
        'hh->?': builtins.int_eq_impl,
        'HH->?': builtins.int_eq_impl,
        'ii->?': builtins.int_eq_impl,
        'II->?': builtins.int_eq_impl,
        'll->?': builtins.int_eq_impl,
        'LL->?': builtins.int_eq_impl,
        'qq->?': builtins.int_eq_impl,
        'QQ->?': builtins.int_eq_impl,
        'ff->?': builtins.real_eq_impl,
        'dd->?': builtins.real_eq_impl,
        'FF->?': npyfuncs.np_complex_eq_impl,
        'DD->?': npyfuncs.np_complex_eq_impl,
    }

    ufunc_db[np.logical_and] = {
        '??->?': npyfuncs.np_logical_and_impl,
        'bb->?': npyfuncs.np_logical_and_impl,
        'BB->?': npyfuncs.np_logical_and_impl,
        'hh->?': npyfuncs.np_logical_and_impl,
        'HH->?': npyfuncs.np_logical_and_impl,
        'ii->?': npyfuncs.np_logical_and_impl,
        'II->?': npyfuncs.np_logical_and_impl,
        'll->?': npyfuncs.np_logical_and_impl,
        'LL->?': npyfuncs.np_logical_and_impl,
        'qq->?': npyfuncs.np_logical_and_impl,
        'QQ->?': npyfuncs.np_logical_and_impl,
        'ff->?': npyfuncs.np_logical_and_impl,
        'dd->?': npyfuncs.np_logical_and_impl,
        'FF->?': npyfuncs.np_complex_logical_and_impl,
        'DD->?': npyfuncs.np_complex_logical_and_impl,
    }

    ufunc_db[np.logical_or] = {
        '??->?': npyfuncs.np_logical_or_impl,
        'bb->?': npyfuncs.np_logical_or_impl,
        'BB->?': npyfuncs.np_logical_or_impl,
        'hh->?': npyfuncs.np_logical_or_impl,
        'HH->?': npyfuncs.np_logical_or_impl,
        'ii->?': npyfuncs.np_logical_or_impl,
        'II->?': npyfuncs.np_logical_or_impl,
        'll->?': npyfuncs.np_logical_or_impl,
        'LL->?': npyfuncs.np_logical_or_impl,
        'qq->?': npyfuncs.np_logical_or_impl,
        'QQ->?': npyfuncs.np_logical_or_impl,
        'ff->?': npyfuncs.np_logical_or_impl,
        'dd->?': npyfuncs.np_logical_or_impl,
        'FF->?': npyfuncs.np_complex_logical_or_impl,
        'DD->?': npyfuncs.np_complex_logical_or_impl,
    }

    ufunc_db[np.logical_xor] = {
        '??->?': npyfuncs.np_logical_xor_impl,
        'bb->?': npyfuncs.np_logical_xor_impl,
        'BB->?': npyfuncs.np_logical_xor_impl,
        'hh->?': npyfuncs.np_logical_xor_impl,
        'HH->?': npyfuncs.np_logical_xor_impl,
        'ii->?': npyfuncs.np_logical_xor_impl,
        'II->?': npyfuncs.np_logical_xor_impl,
        'll->?': npyfuncs.np_logical_xor_impl,
        'LL->?': npyfuncs.np_logical_xor_impl,
        'qq->?': npyfuncs.np_logical_xor_impl,
        'QQ->?': npyfuncs.np_logical_xor_impl,
        'ff->?': npyfuncs.np_logical_xor_impl,
        'dd->?': npyfuncs.np_logical_xor_impl,
        'FF->?': npyfuncs.np_complex_logical_xor_impl,
        'DD->?': npyfuncs.np_complex_logical_xor_impl,
    }

    ufunc_db[np.logical_not] = {
        '?->?': npyfuncs.np_logical_not_impl,
        'b->?': npyfuncs.np_logical_not_impl,
        'B->?': npyfuncs.np_logical_not_impl,
        'h->?': npyfuncs.np_logical_not_impl,
        'H->?': npyfuncs.np_logical_not_impl,
        'i->?': npyfuncs.np_logical_not_impl,
        'I->?': npyfuncs.np_logical_not_impl,
        'l->?': npyfuncs.np_logical_not_impl,
        'L->?': npyfuncs.np_logical_not_impl,
        'q->?': npyfuncs.np_logical_not_impl,
        'Q->?': npyfuncs.np_logical_not_impl,
        'f->?': npyfuncs.np_logical_not_impl,
        'd->?': npyfuncs.np_logical_not_impl,
        'F->?': npyfuncs.np_complex_logical_not_impl,
        'D->?': npyfuncs.np_complex_logical_not_impl,
    }

    ufunc_db[np.maximum] = {
        '??->?': npyfuncs.np_logical_or_impl,
        'bb->b': npyfuncs.np_int_smax_impl,
        'BB->B': npyfuncs.np_int_umax_impl,
        'hh->h': npyfuncs.np_int_smax_impl,
        'HH->H': npyfuncs.np_int_umax_impl,
        'ii->i': npyfuncs.np_int_smax_impl,
        'II->I': npyfuncs.np_int_umax_impl,
        'll->l': npyfuncs.np_int_smax_impl,
        'LL->L': npyfuncs.np_int_umax_impl,
        'qq->q': npyfuncs.np_int_smax_impl,
        'QQ->Q': npyfuncs.np_int_umax_impl,
        'ff->f': npyfuncs.np_real_maximum_impl,
        'dd->d': npyfuncs.np_real_maximum_impl,
        'FF->F': npyfuncs.np_complex_maximum_impl,
        'DD->D': npyfuncs.np_complex_maximum_impl,
    }

    ufunc_db[np.minimum] = {
        '??->?': npyfuncs.np_logical_and_impl,
        'bb->b': npyfuncs.np_int_smin_impl,
        'BB->B': npyfuncs.np_int_umin_impl,
        'hh->h': npyfuncs.np_int_smin_impl,
        'HH->H': npyfuncs.np_int_umin_impl,
        'ii->i': npyfuncs.np_int_smin_impl,
        'II->I': npyfuncs.np_int_umin_impl,
        'll->l': npyfuncs.np_int_smin_impl,
        'LL->L': npyfuncs.np_int_umin_impl,
        'qq->q': npyfuncs.np_int_smin_impl,
        'QQ->Q': npyfuncs.np_int_umin_impl,
        'ff->f': npyfuncs.np_real_minimum_impl,
        'dd->d': npyfuncs.np_real_minimum_impl,
        'FF->F': npyfuncs.np_complex_minimum_impl,
        'DD->D': npyfuncs.np_complex_minimum_impl,
    }

    ufunc_db[np.fmax] = {
        '??->?': npyfuncs.np_logical_or_impl,
        'bb->b': npyfuncs.np_int_smax_impl,
        'BB->B': npyfuncs.np_int_umax_impl,
        'hh->h': npyfuncs.np_int_smax_impl,
        'HH->H': npyfuncs.np_int_umax_impl,
        'ii->i': npyfuncs.np_int_smax_impl,
        'II->I': npyfuncs.np_int_umax_impl,
        'll->l': npyfuncs.np_int_smax_impl,
        'LL->L': npyfuncs.np_int_umax_impl,
        'qq->q': npyfuncs.np_int_smax_impl,
        'QQ->Q': npyfuncs.np_int_umax_impl,
        'ff->f': npyfuncs.np_real_fmax_impl,
        'dd->d': npyfuncs.np_real_fmax_impl,
        'FF->F': npyfuncs.np_complex_fmax_impl,
        'DD->D': npyfuncs.np_complex_fmax_impl,
    }

    ufunc_db[np.fmin] = {
        '??->?': npyfuncs.np_logical_and_impl,
        'bb->b': npyfuncs.np_int_smin_impl,
        'BB->B': npyfuncs.np_int_umin_impl,
        'hh->h': npyfuncs.np_int_smin_impl,
        'HH->H': npyfuncs.np_int_umin_impl,
        'ii->i': npyfuncs.np_int_smin_impl,
        'II->I': npyfuncs.np_int_umin_impl,
        'll->l': npyfuncs.np_int_smin_impl,
        'LL->L': npyfuncs.np_int_umin_impl,
        'qq->q': npyfuncs.np_int_smin_impl,
        'QQ->Q': npyfuncs.np_int_umin_impl,
        'ff->f': npyfuncs.np_real_fmin_impl,
        'dd->d': npyfuncs.np_real_fmin_impl,
        'FF->F': npyfuncs.np_complex_fmin_impl,
        'DD->D': npyfuncs.np_complex_fmin_impl,
    }

    # misc floating functions
    ufunc_db[np.isnan] = {
        'f->?': npyfuncs.np_real_isnan_impl,
        'd->?': npyfuncs.np_real_isnan_impl,
        'F->?': npyfuncs.np_complex_isnan_impl,
        'D->?': npyfuncs.np_complex_isnan_impl,
    }

    ufunc_db[np.isinf] = {
        'f->?': npyfuncs.np_real_isinf_impl,
        'd->?': npyfuncs.np_real_isinf_impl,
        'F->?': npyfuncs.np_complex_isinf_impl,
        'D->?': npyfuncs.np_complex_isinf_impl,
    }

    ufunc_db[np.isfinite] = {
        'f->?': npyfuncs.np_real_isfinite_impl,
        'd->?': npyfuncs.np_real_isfinite_impl,
        'F->?': npyfuncs.np_complex_isfinite_impl,
        'D->?': npyfuncs.np_complex_isfinite_impl,
    }

    ufunc_db[np.signbit] = {
        'f->?': npyfuncs.np_real_signbit_impl,
        'd->?': npyfuncs.np_real_signbit_impl,
    }

    ufunc_db[np.copysign] = {
        'ff->f': npyfuncs.np_real_copysign_impl,
        'dd->d': npyfuncs.np_real_copysign_impl,
    }

    ufunc_db[np.nextafter] = {
        'ff->f': npyfuncs.np_real_nextafter_impl,
        'dd->d': npyfuncs.np_real_nextafter_impl,
    }

    ufunc_db[np.spacing] = {
        'f->f': npyfuncs.np_real_spacing_impl,
        'd->d': npyfuncs.np_real_spacing_impl,
    }

    ufunc_db[np.ldexp] = {
        'fi->f': npyfuncs.np_real_ldexp_impl,
        'fl->f': npyfuncs.np_real_ldexp_impl,
        'di->d': npyfuncs.np_real_ldexp_impl,
        'dl->d': npyfuncs.np_real_ldexp_impl,
    }

    # bit twiddling functions
    ufunc_db[np.bitwise_and] = {
        '??->?': builtins.int_and_impl,
        'bb->b': builtins.int_and_impl,
        'BB->B': builtins.int_and_impl,
        'hh->h': builtins.int_and_impl,
        'HH->H': builtins.int_and_impl,
        'ii->i': builtins.int_and_impl,
        'II->I': builtins.int_and_impl,
        'll->l': builtins.int_and_impl,
        'LL->L': builtins.int_and_impl,
        'qq->q': builtins.int_and_impl,
        'QQ->Q': builtins.int_and_impl,
    }

    ufunc_db[np.bitwise_or] = {
        '??->?': builtins.int_or_impl,
        'bb->b': builtins.int_or_impl,
        'BB->B': builtins.int_or_impl,
        'hh->h': builtins.int_or_impl,
        'HH->H': builtins.int_or_impl,
        'ii->i': builtins.int_or_impl,
        'II->I': builtins.int_or_impl,
        'll->l': builtins.int_or_impl,
        'LL->L': builtins.int_or_impl,
        'qq->q': builtins.int_or_impl,
        'QQ->Q': builtins.int_or_impl,
    }

    ufunc_db[np.bitwise_xor] = {
        '??->?': builtins.int_xor_impl,
        'bb->b': builtins.int_xor_impl,
        'BB->B': builtins.int_xor_impl,
        'hh->h': builtins.int_xor_impl,
        'HH->H': builtins.int_xor_impl,
        'ii->i': builtins.int_xor_impl,
        'II->I': builtins.int_xor_impl,
        'll->l': builtins.int_xor_impl,
        'LL->L': builtins.int_xor_impl,
        'qq->q': builtins.int_xor_impl,
        'QQ->Q': builtins.int_xor_impl,
    }

    ufunc_db[np.invert] = { # aka np.bitwise_not
        '?->?': builtins.bool_invert_impl,
        'b->b': builtins.int_invert_impl,
        'B->B': builtins.int_invert_impl,
        'h->h': builtins.int_invert_impl,
        'H->H': builtins.int_invert_impl,
        'i->i': builtins.int_invert_impl,
        'I->I': builtins.int_invert_impl,
        'l->l': builtins.int_invert_impl,
        'L->L': builtins.int_invert_impl,
        'q->q': builtins.int_invert_impl,
        'Q->Q': builtins.int_invert_impl,
    }

    ufunc_db[np.left_shift] = {
        'bb->b': builtins.int_shl_impl,
        'BB->B': builtins.int_shl_impl,
        'hh->h': builtins.int_shl_impl,
        'HH->H': builtins.int_shl_impl,
        'ii->i': builtins.int_shl_impl,
        'II->I': builtins.int_shl_impl,
        'll->l': builtins.int_shl_impl,
        'LL->L': builtins.int_shl_impl,
        'qq->q': builtins.int_shl_impl,
        'QQ->Q': builtins.int_shl_impl,
    }

    ufunc_db[np.right_shift] = {
        'bb->b': builtins.int_ashr_impl,
        'BB->B': builtins.int_lshr_impl,
        'hh->h': builtins.int_ashr_impl,
        'HH->H': builtins.int_lshr_impl,
        'ii->i': builtins.int_ashr_impl,
        'II->I': builtins.int_lshr_impl,
        'll->l': builtins.int_ashr_impl,
        'LL->L': builtins.int_lshr_impl,
        'qq->q': builtins.int_ashr_impl,
        'QQ->Q': builtins.int_lshr_impl,
    }

    # Inject datetime64 support
    try:
        from . import npdatetime
    except NotImplementedError:
        # Numpy 1.6
        pass
    else:
        ufunc_db[np.negative].update({
            'm->m': npdatetime.timedelta_neg_impl,
            })
        ufunc_db[np.absolute].update({
            'm->m': npdatetime.timedelta_abs_impl,
            })
        ufunc_db[np.sign].update({
            'm->m': npdatetime.timedelta_sign_impl,
            })
        ufunc_db[np.add].update({
            'mm->m': npdatetime.timedelta_add_impl,
            'Mm->M': npdatetime.datetime_plus_timedelta,
            'mM->M': npdatetime.timedelta_plus_datetime,
            })
        ufunc_db[np.subtract].update({
            'mm->m': npdatetime.timedelta_sub_impl,
            'Mm->M': npdatetime.datetime_minus_timedelta,
            'MM->m': npdatetime.datetime_minus_datetime,
            })
        ufunc_db[np.multiply].update({
            'mq->m': npdatetime.timedelta_times_number,
            'md->m': npdatetime.timedelta_times_number,
            'qm->m': npdatetime.number_times_timedelta,
            'dm->m': npdatetime.number_times_timedelta,
            })
        if np.divide != np.true_divide:
            ufunc_db[np.divide].update({
                'mq->m': npdatetime.timedelta_over_number,
                'md->m': npdatetime.timedelta_over_number,
                'mm->d': npdatetime.timedelta_over_timedelta,
            })
        ufunc_db[np.true_divide].update({
            'mq->m': npdatetime.timedelta_over_number,
            'md->m': npdatetime.timedelta_over_number,
            'mm->d': npdatetime.timedelta_over_timedelta,
        })
        ufunc_db[np.floor_divide].update({
            'mq->m': npdatetime.timedelta_over_number,
            'md->m': npdatetime.timedelta_over_number,
        })
        ufunc_db[np.equal].update({
            'MM->?': npdatetime.datetime_eq_datetime_impl,
            'mm->?': npdatetime.timedelta_eq_timedelta_impl,
        })
        ufunc_db[np.not_equal].update({
            'MM->?': npdatetime.datetime_ne_datetime_impl,
            'mm->?': npdatetime.timedelta_ne_timedelta_impl,
        })
        ufunc_db[np.less].update({
            'MM->?': npdatetime.datetime_lt_datetime_impl,
            'mm->?': npdatetime.timedelta_lt_timedelta_impl,
        })
        ufunc_db[np.less_equal].update({
            'MM->?': npdatetime.datetime_le_datetime_impl,
            'mm->?': npdatetime.timedelta_le_timedelta_impl,
        })
        ufunc_db[np.greater].update({
            'MM->?': npdatetime.datetime_gt_datetime_impl,
            'mm->?': npdatetime.timedelta_gt_timedelta_impl,
        })
        ufunc_db[np.greater_equal].update({
            'MM->?': npdatetime.datetime_ge_datetime_impl,
            'mm->?': npdatetime.timedelta_ge_timedelta_impl,
        })
        ufunc_db[np.maximum].update({
            'MM->M': npdatetime.datetime_max_impl,
            'mm->m': npdatetime.timedelta_max_impl,
        })
        ufunc_db[np.minimum].update({
            'MM->M': npdatetime.datetime_min_impl,
            'mm->m': npdatetime.timedelta_min_impl,
        })
        # there is no difference for datetime/timedelta in maximum/fmax
        # and minimum/fmin
        ufunc_db[np.fmax].update({
            'MM->M': npdatetime.datetime_max_impl,
            'mm->m': npdatetime.timedelta_max_impl,
        })
        ufunc_db[np.fmin].update({
            'MM->M': npdatetime.datetime_min_impl,
            'mm->m': npdatetime.timedelta_min_impl,
        })
