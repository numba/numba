"""This file contains information on how to translate different ufuncs
into numba. It is a database of different ufuncs and how each of its
loops maps to a function that implements the inner kernel of that ufunc
(the inner kernel being the per-element function).
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from . import builtins

ufunc_db = {}


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
}
