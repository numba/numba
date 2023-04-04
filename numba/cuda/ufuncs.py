"""Contains information on how to translate different ufuncs for the CUDA
target. It is a database of different ufuncs and how each of its loops maps to
a function that implements the inner kernel of that ufunc (the inner kernel
being the per-element function).

Use get_ufunc_info() to get the information related to a ufunc.
"""

import math
import numpy as np
from functools import lru_cache
from numba.core import typing
from numba.cuda.mathimpl import (get_unary_impl_for_fn_and_ty,
                                 get_binary_impl_for_fn_and_ty)


def get_ufunc_info(ufunc_key):
    return ufunc_db()[ufunc_key]


@lru_cache
def ufunc_db():
    # Imports here are at function scope to avoid circular imports
    from numba.cpython import cmathimpl, mathimpl
    from numba.np.npyfuncs import _check_arity_and_homogeneity
    from numba.np.npyfuncs import (np_complex_acosh_impl, np_complex_cos_impl,
                                   np_complex_sin_impl)

    def np_unary_impl(fn, context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        impl = get_unary_impl_for_fn_and_ty(fn, sig.args[0])
        return impl(context, builder, sig, args)

    def np_binary_impl(fn, context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 2)
        impl = get_binary_impl_for_fn_and_ty(fn, sig.args[0])
        return impl(context, builder, sig, args)

    def np_real_sin_impl(context, builder, sig, args):
        return np_unary_impl(math.sin, context, builder, sig, args)

    def np_real_cos_impl(context, builder, sig, args):
        return np_unary_impl(math.cos, context, builder, sig, args)

    def np_real_tan_impl(context, builder, sig, args):
        return np_unary_impl(math.tan, context, builder, sig, args)

    def np_real_asin_impl(context, builder, sig, args):
        return np_unary_impl(math.asin, context, builder, sig, args)

    def np_real_acos_impl(context, builder, sig, args):
        return np_unary_impl(math.acos, context, builder, sig, args)

    def np_real_atan_impl(context, builder, sig, args):
        return np_unary_impl(math.atan, context, builder, sig, args)

    def np_real_atan2_impl(context, builder, sig, args):
        return np_binary_impl(math.atan2, context, builder, sig, args)

    def np_real_hypot_impl(context, builder, sig, args):
        return np_binary_impl(math.hypot, context, builder, sig, args)

    def np_real_sinh_impl(context, builder, sig, args):
        return np_unary_impl(math.sinh, context, builder, sig, args)

    def np_complex_sinh_impl(context, builder, sig, args):
        # npymath does not provide a complex sinh. The code in funcs.inc.src
        # is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)
        xr = x.real
        xi = x.imag

        sxi = np_real_sin_impl(context, builder, fsig1, [xi])
        shxr = np_real_sinh_impl(context, builder, fsig1, [xr])
        cxi = np_real_cos_impl(context, builder, fsig1, [xi])
        chxr = np_real_cosh_impl(context, builder, fsig1, [xr])

        out.real = builder.fmul(cxi, shxr)
        out.imag = builder.fmul(sxi, chxr)

        return out._getvalue()

    def np_real_cosh_impl(context, builder, sig, args):
        return np_unary_impl(math.cosh, context, builder, sig, args)

    def np_complex_cosh_impl(context, builder, sig, args):
        # npymath does not provide a complex cosh. The code in funcs.inc.src
        # is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)
        xr = x.real
        xi = x.imag

        cxi = np_real_cos_impl(context, builder, fsig1, [xi])
        chxr = np_real_cosh_impl(context, builder, fsig1, [xr])
        sxi = np_real_sin_impl(context, builder, fsig1, [xi])
        shxr = np_real_sinh_impl(context, builder, fsig1, [xr])

        out.real = builder.fmul(cxi, chxr)
        out.imag = builder.fmul(sxi, shxr)

        return out._getvalue()

    def np_real_tanh_impl(context, builder, sig, args):
        return np_unary_impl(math.tanh, context, builder, sig, args)

    def np_complex_tanh_impl(context, builder, sig, args):
        # npymath does not provide complex tan functions. The code
        # in funcs.inc.src for tanh is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        ONE = context.get_constant(fty, 1.0)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)

        xr = x.real
        xi = x.imag
        si = np_real_sin_impl(context, builder, fsig1, [xi])
        ci = np_real_cos_impl(context, builder, fsig1, [xi])
        shr = np_real_sinh_impl(context, builder, fsig1, [xr])
        chr_ = np_real_cosh_impl(context, builder, fsig1, [xr])
        rs = builder.fmul(ci, shr)
        is_ = builder.fmul(si, chr_)
        rc = builder.fmul(ci, chr_)
        # Note: opposite sign for `ic` from code in funcs.inc.src
        ic = builder.fmul(si, shr)
        sqr_rc = builder.fmul(rc, rc)
        sqr_ic = builder.fmul(ic, ic)
        d = builder.fadd(sqr_rc, sqr_ic)
        inv_d = builder.fdiv(ONE, d)
        rs_rc = builder.fmul(rs, rc)
        is_ic = builder.fmul(is_, ic)
        is_rc = builder.fmul(is_, rc)
        rs_ic = builder.fmul(rs, ic)
        numr = builder.fadd(rs_rc, is_ic)
        numi = builder.fsub(is_rc, rs_ic)
        out.real = builder.fmul(numr, inv_d)
        out.imag = builder.fmul(numi, inv_d)

        return out._getvalue()

    def np_real_asinh_impl(context, builder, sig, args):
        return np_unary_impl(math.asinh, context, builder, sig, args)

    def np_real_acosh_impl(context, builder, sig, args):
        return np_unary_impl(math.acosh, context, builder, sig, args)

    def np_real_atanh_impl(context, builder, sig, args):
        return np_unary_impl(math.atanh, context, builder, sig, args)

    db = {}

    db[np.sin] = {
        'f->f': np_real_sin_impl,
        'd->d': np_real_sin_impl,
        'F->F': np_complex_sin_impl,
        'D->D': np_complex_sin_impl,
    }

    db[np.cos] = {
        'f->f': np_real_cos_impl,
        'd->d': np_real_cos_impl,
        'F->F': np_complex_cos_impl,
        'D->D': np_complex_cos_impl,
    }

    db[np.tan] = {
        'f->f': np_real_tan_impl,
        'd->d': np_real_tan_impl,
        'F->F': cmathimpl.tan_impl,
        'D->D': cmathimpl.tan_impl,
    }

    db[np.arcsin] = {
        'f->f': np_real_asin_impl,
        'd->d': np_real_asin_impl,
        'F->F': cmathimpl.asin_impl,
        'D->D': cmathimpl.asin_impl,
    }

    db[np.arccos] = {
        'f->f': np_real_acos_impl,
        'd->d': np_real_acos_impl,
        'F->F': cmathimpl.acos_impl,
        'D->D': cmathimpl.acos_impl,
    }

    db[np.arctan] = {
        'f->f': np_real_atan_impl,
        'd->d': np_real_atan_impl,
        'F->F': cmathimpl.atan_impl,
        'D->D': cmathimpl.atan_impl,
    }

    db[np.arctan2] = {
        'ff->f': np_real_atan2_impl,
        'dd->d': np_real_atan2_impl,
    }

    db[np.hypot] = {
        'ff->f': np_real_hypot_impl,
        'dd->d': np_real_hypot_impl,
    }

    db[np.sinh] = {
        'f->f': np_real_sinh_impl,
        'd->d': np_real_sinh_impl,
        'F->F': np_complex_sinh_impl,
        'D->D': np_complex_sinh_impl,
    }

    db[np.cosh] = {
        'f->f': np_real_cosh_impl,
        'd->d': np_real_cosh_impl,
        'F->F': np_complex_cosh_impl,
        'D->D': np_complex_cosh_impl,
    }

    db[np.tanh] = {
        'f->f': np_real_tanh_impl,
        'd->d': np_real_tanh_impl,
        'F->F': np_complex_tanh_impl,
        'D->D': np_complex_tanh_impl,
    }

    db[np.arcsinh] = {
        'f->f': np_real_asinh_impl,
        'd->d': np_real_asinh_impl,
        'F->F': cmathimpl.asinh_impl,
        'D->D': cmathimpl.asinh_impl,
    }

    db[np.arccosh] = {
        'f->f': np_real_acosh_impl,
        'd->d': np_real_acosh_impl,
        'F->F': np_complex_acosh_impl,
        'D->D': np_complex_acosh_impl,
    }

    db[np.arctanh] = {
        'f->f': np_real_atanh_impl,
        'd->d': np_real_atanh_impl,
        'F->F': cmathimpl.atanh_impl,
        'D->D': cmathimpl.atanh_impl,
    }

    db[np.deg2rad] = {
        'f->f': mathimpl.radians_float_impl,
        'd->d': mathimpl.radians_float_impl,
    }

    db[np.radians] = db[np.deg2rad]

    db[np.rad2deg] = {
        'f->f': mathimpl.degrees_float_impl,
        'd->d': mathimpl.degrees_float_impl,
    }

    db[np.degrees] = db[np.rad2deg]

    return db
