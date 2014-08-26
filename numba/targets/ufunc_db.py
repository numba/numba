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
