"""Codegen for functions used as kernels in NumPy functions

Typically, the kernels of several ufuncs that can't map directly to
Python builtins
"""

from __future__ import print_function, absolute_import, division

import math

from llvmlite.llvmpy import core as lc

from .. import cgutils, typing, types, lowering, errors
from . import mathimpl, numbers

# some NumPy constants. Note that we could generate some of them using
# the math library, but having the values copied from npy_math seems to
# yield more accurate results
_NPY_LOG2E  = 1.442695040888963407359924681001892137 # math.log(math.e, 2)
_NPY_LOG10E = 0.434294481903251827651128918916605082 # math.log(math.e, 10)
_NPY_LOGE2  = 0.693147180559945309417232121458176568 # math.log(2)


def _check_arity_and_homogeneity(sig, args, arity, return_type = None):
    """checks that the following are true:
    - args and sig.args have arg_count elements
    - all input types are homogeneous
    - return type is 'return_type' if provided, otherwise it must be
      homogeneous with the input types.
    """
    assert len(args) == arity
    assert len(sig.args) == arity
    ty = sig.args[0]
    if return_type is None:
        return_type = ty
    # must have homogeneous args
    if not( all(arg==ty for arg in sig.args) and sig.return_type == return_type):
        import inspect
        fname = inspect.currentframe().f_back.f_code.co_name
        msg = '{0} called with invalid types: {1}'.format(fname, sig)
        assert False, msg


def _call_func_by_name_with_cast(context, builder, sig, args,
                                 func_name, ty=types.float64):
    # it is quite common in NumPy to have loops implemented as a call
    # to the double version of the function, wrapped in casts. This
    # helper function facilitates that.
    mod = builder.module
    lty = context.get_argument_type(ty)
    fnty = lc.Type.function(lty, [lty]*len(sig.args))
    fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
    cast_args = [context.cast(builder, arg, argty, ty)
                 for arg, argty in zip(args, sig.args) ]

    result = builder.call(fn, cast_args)
    return context.cast(builder, result, types.float64, sig.return_type)


def _dispatch_func_by_name_type(context, builder, sig, args, table, user_name):
    # for most cases the functions are homogeneous on all their types.
    # this code dispatches on the first argument type as it is the most useful
    # for our uses (all cases but ldexp are homogeneous in all types, and
    # dispatching on the first argument type works of ldexp as well)
    #
    # assumes that the function pointed by func_name has the type
    # signature sig (but needs translation to llvm types).

    ty = sig.args[0]
    try:
        func_name = table[ty]
    except KeyError as e:
        msg = "No {0} function for real type {1}".format(user_name, str(e))
        raise errors.LoweringError(msg)

    mod = builder.module
    if ty in types.complex_domain:
        # In numba struct types are always passed by pointer. So the call has to
        # be transformed from "result = func(ops...)" to "func(&result, ops...).
        # note that the result value pointer as first argument is the convention
        # used by numba.

        # First, prepare the return value
        out = context.make_complex(builder, ty)
        ptrargs = [cgutils.alloca_once_value(builder, arg)
                   for arg in args]
        call_args = [out._getpointer()] + ptrargs
        # get_value_as_argument for struct types like complex allocate stack space
        # and initialize with the value, the return value is the pointer to that
        # allocated space (ie: pointer to a copy of the value in the stack).
        # get_argument_type returns a pointer to the struct type in consonance.
        call_argtys = [ty] + list(sig.args)
        call_argltys = [context.get_value_type(ty).as_pointer()
                        for ty in call_argtys]
        fnty = lc.Type.function(lc.Type.void(), call_argltys)
        # Note: the function isn't pure here (it writes to its pointer args)
        fn = mod.get_or_insert_function(fnty, name=func_name)
        builder.call(fn, call_args)
        retval = builder.load(call_args[0])
    else:
        argtypes = [context.get_argument_type(aty) for aty in sig.args]
        restype = context.get_argument_type(sig.return_type)
        fnty = lc.Type.function(restype, argtypes)
        fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
        retval = context.call_external_function(builder, fn, sig.args, args)
    return retval



########################################################################
# Division kernels inspired by NumPy loops.c.src code
#
# The builtins are not applicable as they rely on a test for zero in the
# denominator. If it is zero the appropriate exception is raised.
# In NumPy, a division by zero does not raise an exception, but instead
# generated a known value. Note that a division by zero in any of the
# operations of a vector may raise an exception or issue a warning
# depending on the np.seterr configuration. This is not supported
# right now (and in any case, it won't be handled by these functions
# either)

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
    with builder.if_else(force_zero, likely=False) as (then, otherwise):
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


def np_int_srem_impl(context, builder, sig, args):
    # based on the actual code in NumPy loops.c.src for signed integers
    _check_arity_and_homogeneity(sig, args, 2)

    num, den = args
    ty = sig.args[0] # any arg type will do, homogeneous
    lty = num.type

    ZERO = context.get_constant(ty, 0)
    den_not_zero = builder.icmp(lc.ICMP_NE, ZERO, den)
    bb_no_if = builder.basic_block
    with cgutils.if_unlikely(builder, den_not_zero):
        bb_if = builder.basic_block
        mod = builder.srem(num,den)
        num_gt_zero = builder.icmp(lc.ICMP_SGT, num, ZERO)
        den_gt_zero = builder.icmp(lc.ICMP_SGT, den, ZERO)
        not_same_sign = builder.xor(num_gt_zero, den_gt_zero)
        mod_not_zero = builder.icmp(lc.ICMP_NE, mod, ZERO)
        needs_fixing = builder.and_(not_same_sign, mod_not_zero)
        fix_value = builder.select(needs_fixing, den, ZERO)
        final_mod = builder.add(fix_value, mod)

    result = builder.phi(lty)
    result.add_incoming(ZERO, bb_no_if)
    result.add_incoming(final_mod, bb_if)

    return result


def np_int_udiv_impl(context, builder, sig, args):
    num, den = args
    lltype = num.type
    assert all(i.type==lltype for i in args), "must have homogeneous types"

    ZERO = lc.Constant.int(lltype, 0)
    div_by_zero = builder.icmp(lc.ICMP_EQ, ZERO, den)
    with builder.if_else(div_by_zero, likely=False) as (then, otherwise):
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


def np_int_urem_impl(context, builder, sig, args):
    # based on the actual code in NumPy loops.c.src for signed integers
    _check_arity_and_homogeneity(sig, args, 2)

    num, den = args
    ty = sig.args[0] # any arg type will do, homogeneous
    lty = num.type

    ZERO = context.get_constant(ty, 0)
    den_not_zero = builder.icmp(lc.ICMP_NE, ZERO, den)
    bb_no_if = builder.basic_block
    with cgutils.if_unlikely(builder, den_not_zero):
        bb_if = builder.basic_block
        mod = builder.srem(num,den)

    result = builder.phi(lty)
    result.add_incoming(ZERO, bb_no_if)
    result.add_incoming(mod, bb_if)

    return result


# implementation of int_fmod is in fact the same as the unsigned remainder,
# that is: srem with a special case returning 0 when the denominator is 0.
np_int_fmod_impl = np_int_urem_impl


def np_real_div_impl(context, builder, sig, args):
    # in NumPy real div has the same semantics as an fdiv for generating
    # NANs, INF and NINF
    _check_arity_and_homogeneity(sig, args, 2)
    return builder.fdiv(*args)


def np_real_mod_impl(context, builder, sig, args):
    # note: this maps to NumPy remainder, which has the same semantics as Python
    # based on code in loops.c.src
    _check_arity_and_homogeneity(sig, args, 2)
    in1, in2 = args
    ty = sig.args[0]

    ZERO = context.get_constant(ty, 0.0)
    res = builder.frem(in1, in2)
    res_ne_zero = builder.fcmp(lc.FCMP_ONE, res, ZERO)
    den_lt_zero = builder.fcmp(lc.FCMP_OLT, in2, ZERO)
    res_lt_zero = builder.fcmp(lc.FCMP_OLT, res, ZERO)
    needs_fixing = builder.and_(res_ne_zero,
                                builder.xor(den_lt_zero, res_lt_zero))
    fix_value = builder.select(needs_fixing, in2, ZERO)

    return builder.fadd(res, fix_value)


def np_real_fmod_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    return builder.frem(*args)


def _fabs(context, builder, arg):
    ZERO = lc.Constant.real(arg.type, 0.0)
    arg_negated = builder.fsub(ZERO, arg)
    arg_is_negative = builder.fcmp(lc.FCMP_OLT, arg, ZERO)
    return builder.select(arg_is_negative, arg_negated, arg)


def np_complex_div_impl(context, builder, sig, args):
    # Extracted from numpy/core/src/umath/loops.c.src,
    # inspired by complex_div_impl
    # variables named coherent with loops.c.src
    # This is implemented using the approach described in
    #   R.L. Smith. Algorithm 116: Complex division.
    #   Communications of the ACM, 5(8):435, 1962

    in1, in2 = [context.make_complex(builder, sig.args[0], value=arg)
                for arg in args]

    in1r = in1.real  # numerator.real
    in1i = in1.imag  # numerator.imag
    in2r = in2.real  # denominator.real
    in2i = in2.imag  # denominator.imag
    ftype = in1r.type
    assert all([i.type==ftype for i in [in1r, in1i, in2r, in2i]]), "mismatched types"
    out = context.make_helper(builder, sig.return_type)

    ZERO = lc.Constant.real(ftype, 0.0)
    ONE = lc.Constant.real(ftype, 1.0)

    # if abs(denominator.real) >= abs(denominator.imag)
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp(lc.FCMP_OGE, in2r_abs, in2i_abs)
    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            # if abs(denominator.real) == 0 and abs(denominator.imag) == 0
            in2r_is_zero = builder.fcmp(lc.FCMP_OEQ, in2r_abs, ZERO)
            in2i_is_zero = builder.fcmp(lc.FCMP_OEQ, in2i_abs, ZERO)
            in2_is_zero = builder.and_(in2r_is_zero, in2i_is_zero)
            with builder.if_else(in2_is_zero) as (inn_then, inn_otherwise):
                with inn_then:
                    # division by 0.
                    # fdiv generates the appropriate NAN/INF/NINF
                    out.real = builder.fdiv(in1r, in2r_abs)
                    out.imag = builder.fdiv(in1i, in2i_abs)
                with inn_otherwise:
                    # general case for:
                    # abs(denominator.real) > abs(denominator.imag)
                    rat = builder.fdiv(in2i, in2r)
                    # scl = 1.0/(in2r + in2i*rat)
                    tmp1 = builder.fmul(in2i, rat)
                    tmp2 = builder.fadd(in2r, tmp1)
                    scl = builder.fdiv(ONE, tmp2)
                    # out.real = (in1r + in1i*rat)*scl
                    # out.imag = (in1i - in1r*rat)*scl
                    tmp3 = builder.fmul(in1i, rat)
                    tmp4 = builder.fmul(in1r, rat)
                    tmp5 = builder.fadd(in1r, tmp3)
                    tmp6 = builder.fsub(in1i, tmp4)
                    out.real = builder.fmul(tmp5, scl)
                    out.imag = builder.fmul(tmp6, scl)
        with otherwise:
            # general case for:
            # abs(denominator.imag) > abs(denominator.real)
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
            out.real = builder.fmul(tmp5, scl)
            out.imag = builder.fmul(tmp6, scl)

    return out._getvalue()


########################################################################
# NumPy logaddexp

def np_real_logaddexp_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.logaddexpf',
        types.float64: 'numba.npymath.logaddexp',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'logaddexp')

########################################################################
# NumPy logaddexp2

def np_real_logaddexp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.logaddexp2f',
        types.float64: 'numba.npymath.logaddexp2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'logaddexp2')


########################################################################
# true div kernels

def np_int_truediv_impl(context, builder, sig, args):
    # in NumPy we don't check for 0 denominator... fdiv handles div by
    # 0 in the way NumPy expects..
    # integer truediv always yields double
    num, den = args
    lltype = num.type
    assert all(i.type==lltype for i in args), "must have homogeneous types"
    numty, denty = sig.args

    num = context.cast(builder, num, numty, types.float64)
    den = context.cast(builder, den, denty, types.float64)

    return builder.fdiv(num,den)


########################################################################
# floor div kernels

def np_real_floor_div_impl(context, builder, sig, args):
    res = np_real_div_impl(context, builder, sig, args)
    s = typing.signature(sig.return_type, sig.return_type)
    return np_real_floor_impl(context, builder, s, (res,))


def np_complex_floor_div_impl(context, builder, sig, args):
    # this is based on the complex floor divide in Numpy's loops.c.src
    # This is basically a full complex division with a complex floor
    # applied.
    # The complex floor seems to be defined as the real floor applied
    # with the real part and zero in the imaginary part. Fully developed
    # so it avoids computing anything related to the imaginary result.
    float_kind = sig.args[0].underlying_float
    floor_sig = typing.signature(float_kind, float_kind)

    in1, in2 = [context.make_complex(builder, sig.args[0], value=arg)
                for arg in args]

    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type==ftype for i in [in1r, in1i, in2r, in2i]]), "mismatched types"

    ZERO = lc.Constant.real(ftype, 0.0)

    out = context.make_helper(builder, sig.return_type)
    out.imag = ZERO

    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp(lc.FCMP_OGE, in2r_abs, in2i_abs)

    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            rat = builder.fdiv(in2i, in2r)
            # out.real = floor((in1r+in1i*rat)/(in2r + in2i*rat))
            tmp1 = builder.fmul(in1i, rat)
            tmp2 = builder.fmul(in2i, rat)
            tmp3 = builder.fadd(in1r, tmp1)
            tmp4 = builder.fadd(in2r, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            # out.real = floor((in1i + in1r*rat)/(in2i + in2r*rat))
            tmp1 = builder.fmul(in1r, rat)
            tmp2 = builder.fmul(in2r, rat)
            tmp3 = builder.fadd(in1i, tmp1)
            tmp4 = builder.fadd(in2i, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
    return out._getvalue()


########################################################################
# numpy power funcs

def np_complex_power_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.complex64: 'numba.npymath.cpowf',
        types.complex128: 'numba.npymath.cpow',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'power')


########################################################################
# Numpy style complex sign

def np_complex_sign_impl(context, builder, sig, args):
    # equivalent to complex sign in NumPy's sign
    # but implemented via selects, balancing the 4 cases.
    _check_arity_and_homogeneity(sig, args, 1)
    op = args[0]
    ty = sig.args[0]
    float_ty = ty.underlying_float

    ZERO = context.get_constant(float_ty, 0.0)
    ONE  = context.get_constant(float_ty, 1.0)
    MINUS_ONE = context.get_constant(float_ty, -1.0)
    NAN = context.get_constant(float_ty, float('nan'))
    result = context.make_complex(builder, ty)
    result.real = ZERO
    result.imag = ZERO

    cmp_sig = typing.signature(types.boolean, *[ty] * 2)
    cmp_args = [op, result._getvalue()]
    arg1_ge_arg2 = np_complex_ge_impl(context, builder, cmp_sig, cmp_args)
    arg1_eq_arg2 = np_complex_eq_impl(context, builder, cmp_sig, cmp_args)
    arg1_lt_arg2 = np_complex_lt_impl(context, builder, cmp_sig, cmp_args)

    real_when_ge = builder.select(arg1_eq_arg2, ZERO, ONE)
    real_when_nge = builder.select(arg1_lt_arg2, MINUS_ONE, NAN)
    result.real = builder.select(arg1_ge_arg2, real_when_ge, real_when_nge)

    return result._getvalue()


########################################################################
# Numpy rint

def np_real_rint_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    return mathimpl.call_fp_intrinsic(builder, 'llvm.rint', args)


def np_complex_rint_impl(context, builder, sig, args):
    # based on code in NumPy's funcs.inc.src
    # rint of a complex number defined as rint of its real and imag
    # parts
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)

    inner_sig = typing.signature(*[float_ty]*2)
    out.real = np_real_rint_impl(context, builder, inner_sig, [in1.real])
    out.imag = np_real_rint_impl(context, builder, inner_sig, [in1.imag])
    return out._getvalue()


########################################################################
# NumPy exp

def np_real_exp_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.expf',
        types.float64: 'numba.npymath.exp',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp')


def np_complex_exp_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.cexpf',
        types.complex128: 'numba.npymath.cexp',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp')

########################################################################
# NumPy exp2

def np_real_exp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.exp2f',
        types.float64: 'numba.npymath.exp2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp2')


def np_complex_exp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    in1 = context.make_complex(builder, ty, value=args[0])
    tmp = context.make_complex(builder, ty)
    loge2 = context.get_constant(float_ty, _NPY_LOGE2)
    tmp.real = builder.fmul(loge2, in1.real)
    tmp.imag = builder.fmul(loge2, in1.imag)
    return np_complex_exp_impl(context, builder, sig, [tmp._getvalue()])


########################################################################
# NumPy log

def np_real_log_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.logf',
        types.float64: 'numba.npymath.log',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log')


def np_complex_log_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.clogf',
        types.complex128: 'numba.npymath.clog',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log')

########################################################################
# NumPy log2

def np_real_log2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.log2f',
        types.float64: 'numba.npymath.log2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log2')

def np_complex_log2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = context.make_complex(builder, ty, value=tmp)
    log2e = context.get_constant(float_ty, _NPY_LOG2E)
    tmp.real = builder.fmul(log2e, tmp.real)
    tmp.imag = builder.fmul(log2e, tmp.imag)
    return tmp._getvalue()


########################################################################
# NumPy log10

def np_real_log10_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.log10f',
        types.float64: 'numba.npymath.log10',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log10')


def np_complex_log10_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = context.make_complex(builder, ty, value=tmp)
    log10e = context.get_constant(float_ty, _NPY_LOG10E)
    tmp.real = builder.fmul(log10e, tmp.real)
    tmp.imag = builder.fmul(log10e, tmp.imag)
    return tmp._getvalue()


########################################################################
# NumPy expm1

def np_real_expm1_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.expm1f',
        types.float64: 'numba.npymath.expm1',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'expm1')

def np_complex_expm1_impl(context, builder, sig, args):
    # this is based on nc_expm1 in funcs.inc.src
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty]*2)

    MINUS_ONE = context.get_constant(float_ty, -1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    a = np_real_exp_impl(context, builder, float_unary_sig, [in1.real])
    out = context.make_complex(builder, ty)
    cos_imag = np_real_cos_impl(context, builder, float_unary_sig, [in1.imag])
    sin_imag = np_real_sin_impl(context, builder, float_unary_sig, [in1.imag])
    tmp = builder.fmul(a, cos_imag)
    out.imag = builder.fmul(a, sin_imag)
    out.real = builder.fadd(tmp, MINUS_ONE)

    return out._getvalue()


########################################################################
# NumPy log1p

def np_real_log1p_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.log1pf',
        types.float64: 'numba.npymath.log1p',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log1p')

def np_complex_log1p_impl(context, builder, sig, args):
    # base on NumPy's nc_log1p in funcs.inc.src
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty]*2)
    float_binary_sig = typing.signature(*[float_ty]*3)

    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    real_plus_one = builder.fadd(in1.real, ONE)
    l = np_real_hypot_impl(context, builder, float_binary_sig,
                           [real_plus_one, in1.imag])
    out.imag = np_real_atan2_impl(context, builder, float_binary_sig,
                                  [in1.imag, real_plus_one])
    out.real = np_real_log_impl(context, builder, float_unary_sig, [l])

    return out._getvalue()


########################################################################
# NumPy sqrt

def np_real_sqrt_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.sqrtf',
        types.float64: 'numba.npymath.sqrt',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sqrt')


def np_complex_sqrt_impl(context, builder, sig, args):
    dispatch_table = {
        types.complex64: 'numba.npymath.csqrtf',
        types.complex128: 'numba.npymath.csqrt',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sqrt')


########################################################################
# NumPy square

def np_int_square_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    return builder.mul(args[0], args[0])


def np_real_square_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    return builder.fmul(args[0], args[0])

def np_complex_square_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    binary_sig = typing.signature(*[sig.return_type]*3)
    return numbers.complex_mul_impl(context, builder, binary_sig,
                                     [args[0], args[0]])


########################################################################
# NumPy reciprocal

def np_int_reciprocal_impl(context, builder, sig, args):
    # based on the implementation in loops.c.src
    # integer versions for reciprocal are performed via promotion
    # using double, and then converted back to the type
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.return_type

    binary_sig = typing.signature(*[ty]*3)
    in_as_float = context.cast(builder, args[0], ty, types.float64)
    ONE = context.get_constant(types.float64, 1)
    result_as_float = builder.fdiv(ONE, in_as_float)
    return context.cast(builder, result_as_float, types.float64, ty)


def np_real_reciprocal_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ONE = context.get_constant(sig.return_type, 1.0)
    return builder.fdiv(ONE, args[0])


def np_complex_reciprocal_impl(context, builder, sig, args):
    # based on the implementation in loops.c.src
    # Basically the same Smith method used for division, but with
    # the numerator substitued by 1.0
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float

    ZERO = context.get_constant(float_ty, 0.0)
    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    in1r = in1.real
    in1i = in1.imag
    in1r_abs = _fabs(context, builder, in1r)
    in1i_abs = _fabs(context, builder, in1i)
    in1i_abs_le_in1r_abs = builder.fcmp(lc.FCMP_OLE, in1i_abs, in1r_abs)

    with builder.if_else(in1i_abs_le_in1r_abs) as (then, otherwise):
        with then:
            r = builder.fdiv(in1i, in1r)
            tmp0 = builder.fmul(in1i, r)
            d = builder.fadd(in1r, tmp0)
            inv_d = builder.fdiv(ONE, d)
            minus_r = builder.fsub(ZERO, r)
            out.real = inv_d
            out.imag = builder.fmul(minus_r, inv_d)
        with otherwise:
            r = builder.fdiv(in1r, in1i)
            tmp0 = builder.fmul(in1r, r)
            d = builder.fadd(tmp0, in1i)
            inv_d = builder.fdiv(ONE, d)
            out.real = builder.fmul(r, inv_d)
            out.imag = builder.fsub(ZERO, inv_d)

    return out._getvalue()


########################################################################
# NumPy sin

def np_real_sin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.sinf',
        types.float64: 'numba.npymath.sin',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sin')


def np_complex_sin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.csinf',
        types.complex128: 'numba.npymath.csin',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sin')


########################################################################
# NumPy cos

def np_real_cos_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.cosf',
        types.float64: 'numba.npymath.cos',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'cos')


def np_complex_cos_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.ccosf',
        types.complex128: 'numba.npymath.ccos',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'cos')


########################################################################
# NumPy tan

def np_real_tan_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.tanf',
        types.float64: 'numba.npymath.tan',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'tan')


def np_complex_tan_impl(context, builder, sig, args):
    # npymath does not provide complex tan functions. The code
    # in funcs.inc.src for tan is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty]*2)
    ONE = context.get_constant(float_ty, 1.0)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)

    xr = x.real
    xi = x.imag
    sr = np_real_sin_impl(context, builder, float_unary_sig, [xr])
    cr = np_real_cos_impl(context, builder, float_unary_sig, [xr])
    shi = np_real_sinh_impl(context, builder, float_unary_sig, [xi])
    chi = np_real_cosh_impl(context, builder, float_unary_sig, [xi])
    rs = builder.fmul(sr, chi)
    is_ = builder.fmul(cr, shi)
    rc = builder.fmul(cr, chi)
    ic = builder.fmul(sr, shi) # note: opposite sign from code in funcs.inc.src
    sqr_rc = builder.fmul(rc, rc)
    sqr_ic = builder.fmul(ic, ic)
    d = builder.fadd(sqr_rc, sqr_ic)
    inv_d = builder.fdiv(ONE, d)
    rs_rc = builder.fmul(rs, rc)
    is_ic = builder.fmul(is_, ic)
    is_rc = builder.fmul(is_, rc)
    rs_ic = builder.fmul(rs, ic)
    numr = builder.fsub(rs_rc, is_ic)
    numi = builder.fadd(is_rc, rs_ic)
    out.real = builder.fmul(numr, inv_d)
    out.imag = builder.fmul(numi, inv_d)

    return out._getvalue()


########################################################################
# NumPy asin

def np_real_asin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.asinf',
        types.float64: 'numba.npymath.asin',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arcsin')


def _complex_expand_series(context, builder, ty, initial, x, coefs):
    """this is used to implement approximations using series that are
    quite common in NumPy's source code to improve precision when the
    magnitude of the arguments is small. In funcs.inc.src this is
    implemented by repeated use of the macro "SERIES_HORNER_TERM
    """
    assert ty in types.complex_domain
    binary_sig = typing.signature(*[ty]*3)
    accum = context.make_complex(builder, ty, value=initial)
    ONE = context.get_constant(ty.underlying_float, 1.0)
    for coef in reversed(coefs):
        constant = context.get_constant(ty.underlying_float, coef)
        value = numbers.complex_mul_impl(context, builder, binary_sig,
                                          [x, accum._getvalue()])
        accum._setvalue(value)
        accum.real = builder.fadd(ONE, builder.fmul(accum.real, constant))
        accum.imag = builder.fmul(accum.imag, constant)

    return accum._getvalue()


def np_complex_asin_impl(context, builder, sig, args):
    # npymath does not provide a complex asin. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    epsilon = context.get_constant(float_ty, 1e-3)

    # if real or imag has magnitude over 1e-3...
    x = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    abs_r = _fabs(context, builder, x.real)
    abs_i = _fabs(context, builder, x.imag)
    abs_r_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_r, epsilon)
    abs_i_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_i, epsilon)
    any_gt_epsilon = builder.or_(abs_r_gt_epsilon, abs_i_gt_epsilon)
    complex_binary_sig = typing.signature(*[ty]*3)
    with builder.if_else(any_gt_epsilon) as (then, otherwise):
        with then:
            # ... then use formula:
            # - j * log(j * x + sqrt(1 - sqr(x)))
            I = context.get_constant_generic(builder, ty, 1.0j)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            ZERO = context.get_constant_generic(builder, ty, 0.0 + 0.0j)
            xx = np_complex_square_impl(context, builder, sig, args)
            one_minus_xx = numbers.complex_sub_impl(context, builder,
                                                     complex_binary_sig,
                                                     [ONE, xx])
            sqrt_one_minus_xx = np_complex_sqrt_impl(context, builder, sig,
                                                     [one_minus_xx])
            ix = numbers.complex_mul_impl(context, builder,
                                           complex_binary_sig,
                                           [I, args[0]])
            log_arg = numbers.complex_add_impl(context, builder, sig,
                                                [ix, sqrt_one_minus_xx])
            log = np_complex_log_impl(context, builder, sig, [log_arg])
            ilog = numbers.complex_mul_impl(context, builder,
                                             complex_binary_sig,
                                             [I, log])
            out._setvalue(numbers.complex_sub_impl(context, builder,
                                                    complex_binary_sig,
                                                    [ZERO, ilog]))
        with otherwise:
            # ... else use series expansion (to avoid loss of precision)
            coef_dict = {
                types.complex64: [1.0/6.0, 9.0/20.0],
                types.complex128: [1.0/6.0, 9.0/20.0, 25.0/42.0],
                # types.complex256: [1.0/6.0, 9.0/20.0, 25.0/42.0, 49.0/72.0, 81.0/110.0]
            }

            xx = np_complex_square_impl(context, builder, sig, args)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            tmp = _complex_expand_series(context, builder, ty,
                                         ONE, xx, coef_dict[ty])
            out._setvalue(numbers.complex_mul_impl(context, builder,
                                                    complex_binary_sig,
                                                    [args[0], tmp]))

    return out._getvalue()


########################################################################
# NumPy acos

def np_real_acos_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.acosf',
        types.float64: 'numba.npymath.acos',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arccos')


########################################################################
# NumPy atan

def np_real_atan_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.atanf',
        types.float64: 'numba.npymath.atan',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arctan')


def np_complex_atan_impl(context, builder, sig, args):
    # npymath does not provide a complex atan. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    epsilon = context.get_constant(float_ty, 1e-3)

    # if real or imag has magnitude over 1e-3...
    x = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    abs_r = _fabs(context, builder, x.real)
    abs_i = _fabs(context, builder, x.imag)
    abs_r_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_r, epsilon)
    abs_i_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_i, epsilon)
    any_gt_epsilon = builder.or_(abs_r_gt_epsilon, abs_i_gt_epsilon)
    binary_sig = typing.signature(*[ty]*3)
    with builder.if_else(any_gt_epsilon) as (then, otherwise):
        with then:
            # ... then use formula
            # 0.5j * log((j + x)/(j - x))
            I = context.get_constant_generic(builder, ty, 0.0 + 1.0j)
            I2 = context.get_constant_generic(builder, ty, 0.0 + 0.5j)
            den = numbers.complex_sub_impl(context, builder, binary_sig,
                                            [I, args[0]])
            num = numbers.complex_add_impl(context, builder, binary_sig,
                                            [I, args[0]])
            div = np_complex_div_impl(context, builder, binary_sig,
                                      [num, den])
            log = np_complex_log_impl(context, builder, sig, [div])
            res = numbers.complex_mul_impl(context, builder, binary_sig,
                                            [I2, log])

            out._setvalue(res)
        with otherwise:
            # else use series expansion (to avoid loss of precision)
            coef_dict = {
                types.complex64: [-1.0/3.0, -3.0/5.0],
                types.complex128: [-1.0/3.0, -3.0/5.0, -5.0/7.0],
                # types.complex256: [-1.0/3.0, -3.0/5.0, -5.0/7.0, -7.0/9.0, -9.0/11.0]
            }

            xx = np_complex_square_impl(context, builder, sig, args)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            tmp = _complex_expand_series(context, builder, ty,
                                         ONE, xx, coef_dict[ty])
            out._setvalue(numbers.complex_mul_impl(context, builder,
                                                    binary_sig,
                                                    [args[0], tmp]))

    return out._getvalue()


########################################################################
# NumPy atan2

def np_real_atan2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.atan2f',
        types.float64: 'numba.npymath.atan2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arctan2')


########################################################################
# NumPy hypot

def np_real_hypot_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.hypotf',
        types.float64: 'numba.npymath.hypot',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'hypot')


########################################################################
# NumPy sinh

def np_real_sinh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.sinhf',
        types.float64: 'numba.npymath.sinh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sinh')


def np_complex_sinh_impl(context, builder, sig, args):
    # npymath does not provide a complex sinh. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)


    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty]*2)
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


########################################################################
# NumPy cosh

def np_real_cosh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.coshf',
        types.float64: 'numba.npymath.cosh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'cosh')


def np_complex_cosh_impl(context, builder, sig, args):
    # npymath does not provide a complex cosh. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty]*2)
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


########################################################################
# NumPy tanh

def np_real_tanh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.tanhf',
        types.float64: 'numba.npymath.tanh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'tanh')


def np_complex_tanh_impl(context, builder, sig, args):
    # npymath does not provide complex tan functions. The code
    # in funcs.inc.src for tanh is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty]*2)
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
    ic = builder.fmul(si, shr) # note: opposite sign from code in funcs.inc.src
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


########################################################################
# NumPy asinh

def np_real_asinh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.asinhf',
        types.float64: 'numba.npymath.asinh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arcsinh')


def np_complex_asinh_impl(context, builder, sig, args):
    # npymath does not provide a complex atan. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    epsilon = context.get_constant(float_ty, 1e-3)

    # if real or imag has magnitude over 1e-3...
    x = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    abs_r = _fabs(context, builder, x.real)
    abs_i = _fabs(context, builder, x.imag)
    abs_r_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_r, epsilon)
    abs_i_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_i, epsilon)
    any_gt_epsilon = builder.or_(abs_r_gt_epsilon, abs_i_gt_epsilon)
    binary_sig = typing.signature(*[ty]*3)
    with builder.if_else(any_gt_epsilon) as (then, otherwise):
        with then:
            # ... then use formula
            # log(sqrt(1+sqr(x)) + x)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            xx = np_complex_square_impl(context, builder, sig, args)
            one_plus_xx = numbers.complex_add_impl(context, builder,
                                                    binary_sig, [ONE, xx])
            sqrt_res = np_complex_sqrt_impl(context, builder, sig,
                                            [one_plus_xx])
            log_arg = numbers.complex_add_impl(context, builder,
                                                binary_sig, [sqrt_res, args[0]])
            res = np_complex_log_impl(context, builder, sig, [log_arg])
            out._setvalue(res)
        with otherwise:
            # else use series expansion (to avoid loss of precision)
            coef_dict = {
                types.complex64: [-1.0/6.0, -9.0/20.0],
                types.complex128: [-1.0/6.0, -9.0/20.0, -25.0/42.0],
                # types.complex256: [-1.0/6.0, -9.0/20.0, -25.0/42.0, -49.0/72.0, -81.0/110.0]
            }

            xx = np_complex_square_impl(context, builder, sig, args)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            tmp = _complex_expand_series(context, builder, ty,
                                         ONE, xx, coef_dict[ty])
            out._setvalue(numbers.complex_mul_impl(context, builder,
                                                    binary_sig,
                                                    [args[0], tmp]))

    return out._getvalue()


########################################################################
# NumPy acosh

def np_real_acosh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.acoshf',
        types.float64: 'numba.npymath.acosh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arccosh')


def np_complex_acosh_impl(context, builder, sig, args):
    # npymath does not provide a complex acosh. The code in funcs.inc.src
    # is translated here...
    # log(x + sqrt(x+1) * sqrt(x-1))
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    csig2 = typing.signature(*[ty]*3)

    ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
    x = args[0]

    x_plus_one = numbers.complex_add_impl(context, builder, csig2, [x,
                                                                     ONE])
    x_minus_one = numbers.complex_sub_impl(context, builder, csig2, [x,
                                                                      ONE])
    sqrt_x_plus_one = np_complex_sqrt_impl(context, builder, sig, [x_plus_one])
    sqrt_x_minus_one = np_complex_sqrt_impl(context, builder, sig, [x_minus_one])
    prod_sqrt = numbers.complex_mul_impl(context, builder, csig2,
                                          [sqrt_x_plus_one,
                                           sqrt_x_minus_one])
    log_arg = numbers.complex_add_impl(context, builder, csig2, [x,
                                                                  prod_sqrt])

    return np_complex_log_impl(context, builder, sig, [log_arg])


########################################################################
# NumPy atanh

def np_real_atanh_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.atanhf',
        types.float64: 'numba.npymath.atanh',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'arctanh')


def np_complex_atanh_impl(context, builder, sig, args):
    # npymath does not provide a complex atanh. The code in funcs.inc.src
    # is translated here...
    _check_arity_and_homogeneity(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    epsilon = context.get_constant(float_ty, 1e-3)

    # if real or imag has magnitude over 1e-3...
    x = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    abs_r = _fabs(context, builder, x.real)
    abs_i = _fabs(context, builder, x.imag)
    abs_r_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_r, epsilon)
    abs_i_gt_epsilon = builder.fcmp(lc.FCMP_OGT, abs_i, epsilon)
    any_gt_epsilon = builder.or_(abs_r_gt_epsilon, abs_i_gt_epsilon)
    binary_sig = typing.signature(*[ty]*3)
    with builder.if_else(any_gt_epsilon) as (then, otherwise):
        with then:
            # ... then use formula
            # 0.5 * log((1 + x)/(1 - x))
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            HALF = context.get_constant_generic(builder, ty, 0.5 + 0.0j)
            den = numbers.complex_sub_impl(context, builder, binary_sig,
                                            [ONE, args[0]])
            num = numbers.complex_add_impl(context, builder, binary_sig,
                                            [ONE, args[0]])
            div = np_complex_div_impl(context, builder, binary_sig,
                                      [num, den])
            log = np_complex_log_impl(context, builder, sig, [div])
            res = numbers.complex_mul_impl(context, builder, binary_sig,
                                            [HALF, log])

            out._setvalue(res)
        with otherwise:
            # else use series expansion (to avoid loss of precision)
            coef_dict = {
                types.complex64: [1.0/3.0, 3.0/5.0],
                types.complex128: [1.0/3.0, 3.0/5.0, 5.0/7.0],
                # types.complex256: [1.0/3.0, 3.0/5.0, 5.0/7.0, 7.0/9.0, 9.0/11.0]
            }

            xx = np_complex_square_impl(context, builder, sig, args)
            ONE = context.get_constant_generic(builder, ty, 1.0 + 0.0j)
            tmp = _complex_expand_series(context, builder, ty,
                                         ONE, xx, coef_dict[ty])
            out._setvalue(numbers.complex_mul_impl(context, builder,
                                                    binary_sig,
                                                    [args[0], tmp]))

    return out._getvalue()



########################################################################
# NumPy floor

def np_real_floor_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    return mathimpl.call_fp_intrinsic(builder, 'llvm.floor', args)


########################################################################
# NumPy ceil

def np_real_ceil_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    return mathimpl.call_fp_intrinsic(builder, 'llvm.ceil', args)


########################################################################
# NumPy trunc

def np_real_trunc_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    return mathimpl.call_fp_intrinsic(builder, 'llvm.trunc', args)


########################################################################
# NumPy fabs

def np_real_fabs_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    return mathimpl.call_fp_intrinsic(builder, 'llvm.fabs', args)


########################################################################
# NumPy style predicates

# For real and integer types rely on numbers... but complex ordering in
# NumPy is lexicographic (while Python does not provide ordering).
def np_complex_ge_impl(context, builder, sig, args):
    # equivalent to macro CGE in NumPy's loops.c.src
    # ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi >= yi))
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_gt_yr = builder.fcmp(lc.FCMP_OGT, xr, yr)
    no_nan_xi_yi = builder.fcmp(lc.FCMP_ORD, xi, yi)
    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_ge_yi = builder.fcmp(lc.FCMP_OGE, xi, yi)
    first_term = builder.and_(xr_gt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_ge_yi)
    return builder.or_(first_term, second_term)


def np_complex_le_impl(context, builder, sig, args):
    # equivalent to macro CLE in NumPy's loops.c.src
    # ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi <= yi))
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_lt_yr = builder.fcmp(lc.FCMP_OLT, xr, yr)
    no_nan_xi_yi = builder.fcmp(lc.FCMP_ORD, xi, yi)
    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_le_yi = builder.fcmp(lc.FCMP_OLE, xi, yi)
    first_term = builder.and_(xr_lt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_le_yi)
    return builder.or_(first_term, second_term)


def np_complex_gt_impl(context, builder, sig, args):
    # equivalent to macro CGT in NumPy's loops.c.src
    # ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi > yi))
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_gt_yr = builder.fcmp(lc.FCMP_OGT, xr, yr)
    no_nan_xi_yi = builder.fcmp(lc.FCMP_ORD, xi, yi)
    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_gt_yi = builder.fcmp(lc.FCMP_OGT, xi, yi)
    first_term = builder.and_(xr_gt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_gt_yi)
    return builder.or_(first_term, second_term)


def np_complex_lt_impl(context, builder, sig, args):
    # equivalent to macro CLT in NumPy's loops.c.src
    # ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi < yi))
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_lt_yr = builder.fcmp(lc.FCMP_OLT, xr, yr)
    no_nan_xi_yi = builder.fcmp(lc.FCMP_ORD, xi, yi)
    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_lt_yi = builder.fcmp(lc.FCMP_OLT, xi, yi)
    first_term = builder.and_(xr_lt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_lt_yi)
    return builder.or_(first_term, second_term)


def np_complex_eq_impl(context, builder, sig, args):
    # equivalent to macro CEQ in NumPy's loops.c.src
    # (xr == yr && xi == yi)
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_eq_yi = builder.fcmp(lc.FCMP_OEQ, xi, yi)
    return builder.and_(xr_eq_yr, xi_eq_yi)


def np_complex_ne_impl(context, builder, sig, args):
    # equivalent to macro CNE in NumPy's loops.c.src
    # (xr != yr || xi != yi)
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)

    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_ne_yr = builder.fcmp(lc.FCMP_UNE, xr, yr)
    xi_ne_yi = builder.fcmp(lc.FCMP_UNE, xi, yi)
    return builder.or_(xr_ne_yr, xi_ne_yi)


########################################################################
# NumPy logical algebra

# these are made generic for all types for now, assuming that
# cgutils.is_true works in the underlying types.

def _complex_is_true(context, builder, ty, val):
    complex_val = context.make_complex(builder, ty, value=val)
    re_true = cgutils.is_true(builder, complex_val.real)
    im_true = cgutils.is_true(builder, complex_val.imag)
    return builder.or_(re_true, im_true)


def np_logical_and_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.and_(a, b)


def np_complex_logical_and_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.and_(a, b)


def np_logical_or_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.or_(a, b)


def np_complex_logical_or_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.or_(a, b)


def np_logical_xor_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.xor(a, b)


def np_complex_logical_xor_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.xor(a, b)


def np_logical_not_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return cgutils.is_false(builder, args[0])


def np_complex_logical_not_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    return builder.not_(a)

########################################################################
# NumPy style max/min
#
# There are 2 different sets of functions to perform max and min in
# NumPy: maximum/minimum and fmax/fmin.
# Both differ in the way NaNs are handled, so the actual differences
# come in action only on float/complex numbers. The functions used for
# integers is shared. For booleans maximum is equivalent to or, and
# minimum is equivalent to and. Datetime support will go elsewhere.

def np_int_smax_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_sge_arg2 = builder.icmp(lc.ICMP_SGE, arg1, arg2)
    return builder.select(arg1_sge_arg2, arg1, arg2)


def np_int_umax_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_uge_arg2 = builder.icmp(lc.ICMP_UGE, arg1, arg2)
    return builder.select(arg1_uge_arg2, arg1, arg2)


def np_real_maximum_impl(context, builder, sig, args):
    # maximum prefers nan (tries to return a nan).
    _check_arity_and_homogeneity(sig, args, 2)

    arg1, arg2 = args
    arg1_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg1)
    any_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg2)
    nan_result = builder.select(arg1_nan, arg1, arg2)

    arg1_ge_arg2 = builder.fcmp(lc.FCMP_OGE, arg1, arg2)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_real_fmax_impl(context, builder, sig, args):
    # fmax prefers non-nan (tries to return a non-nan).
    _check_arity_and_homogeneity(sig, args, 2)

    arg1, arg2 = args
    arg2_nan = builder.fcmp(lc.FCMP_UNO, arg2, arg2)
    any_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg2)
    nan_result = builder.select(arg2_nan, arg1, arg2)

    arg1_ge_arg2 = builder.fcmp(lc.FCMP_OGE, arg1, arg2)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_complex_maximum_impl(context, builder, sig, args):
    # maximum prefers nan (tries to return a nan).
    # There is an extra caveat with complex numbers, as there is more
    # than one type of nan. NumPy's docs state that the nan in the
    # first argument is returned when both arguments are nans.
    # If only one nan is found, that nan is returned.
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty]*2)
    arg1, arg2 = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg1_nan, arg1, arg2)

    arg1_ge_arg2 = np_complex_ge_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_complex_fmax_impl(context, builder, sig, args):
    # fmax prefers non-nan (tries to return a non-nan).
    # There is an extra caveat with complex numbers, as there is more
    # than one type of nan. NumPy's docs state that the nan in the
    # first argument is returned when both arguments are nans.
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty]*2)
    arg1, arg2 = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg2_nan, arg1, arg2)

    arg1_ge_arg2 = np_complex_ge_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_int_smin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_sle_arg2 = builder.icmp(lc.ICMP_SLE, arg1, arg2)
    return builder.select(arg1_sle_arg2, arg1, arg2)


def np_int_umin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_ule_arg2 = builder.icmp(lc.ICMP_ULE, arg1, arg2)
    return builder.select(arg1_ule_arg2, arg1, arg2)


def np_real_minimum_impl(context, builder, sig, args):
    # minimum prefers nan (tries to return a nan).
    _check_arity_and_homogeneity(sig, args, 2)

    arg1, arg2 = args
    arg1_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg1)
    any_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg2)
    nan_result = builder.select(arg1_nan, arg1, arg2)

    arg1_le_arg2 = builder.fcmp(lc.FCMP_OLE, arg1, arg2)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_real_fmin_impl(context, builder, sig, args):
    # fmin prefers non-nan (tries to return a non-nan).
    _check_arity_and_homogeneity(sig, args, 2)

    arg1, arg2 = args
    arg1_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg1)
    any_nan = builder.fcmp(lc.FCMP_UNO, arg1, arg2)
    nan_result = builder.select(arg1_nan, arg2, arg1)

    arg1_le_arg2 = builder.fcmp(lc.FCMP_OLE, arg1, arg2)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_complex_minimum_impl(context, builder, sig, args):
    # minimum prefers nan (tries to return a nan).
    # There is an extra caveat with complex numbers, as there is more
    # than one type of nan. NumPy's docs state that the nan in the
    # first argument is returned when both arguments are nans.
    # If only one nan is found, that nan is returned.
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty]*2)
    arg1, arg2 = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg1_nan, arg1, arg2)

    arg1_le_arg2 = np_complex_le_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


def np_complex_fmin_impl(context, builder, sig, args):
    # fmin prefers non-nan (tries to return a non-nan).
    # There is an extra caveat with complex numbers, as there is more
    # than one type of nan. NumPy's docs state that the nan in the
    # first argument is returned when both arguments are nans.
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty]*2)
    arg1, arg2 = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg2_nan, arg1, arg2)

    arg1_le_arg2 = np_complex_le_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)

    return builder.select(any_nan, nan_result, non_nan_result)


########################################################################
# NumPy floating point misc

def np_real_isnan_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    x, = args

    return builder.fcmp(lc.FCMP_UNO, x, x)


def np_complex_isnan_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)

    x, = args
    ty, = sig.args
    complex_val = context.make_complex(builder, ty, value=x)

    return builder.fcmp(lc.FCMP_UNO, complex_val.real, complex_val.imag)


def _real_is_not_finite(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    x, = args
    ty, = sig.args
    f_ff_sig = typing.signature(*[ty]*3)
    sub = numbers.real_sub_impl(context, builder, f_ff_sig, [x, x])

    return np_real_isnan_impl(context, builder, sig, [sub])


def np_real_isfinite_impl(context, builder, sig, args):
    # in NumPy, when there is not an appropriate builtin, it falls back to
    # ! npy_isnan((x) + (-x)). Use this code to avoid having to add a call.
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)

    return builder.not_(_real_is_not_finite(context, builder, sig, args))


def np_complex_isfinite_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    x, = args
    ty, = sig.args
    fty = ty.underlying_float
    b_f_sig = typing.signature(types.boolean, fty)
    complex_val = context.make_complex(builder, ty, value=x)
    real_isfinite = np_real_isfinite_impl(context, builder, b_f_sig,
                                          [complex_val.real])
    imag_isfinite = np_real_isfinite_impl(context, builder, b_f_sig,
                                          [complex_val.imag])

    return builder.and_(real_isfinite, imag_isfinite)


def np_real_isinf_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    x, = args
    not_finite = _real_is_not_finite(context, builder, sig, args)
    not_nan = builder.fcmp(lc.FCMP_ORD, x, x)

    return builder.and_(not_finite, not_nan)


def np_complex_isinf_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    ty, = sig.args
    fty = ty.underlying_float
    b_f_sig = typing.signature(types.boolean, fty)
    x = context.make_complex(builder, ty, value=args[0])
    real_isinf = np_real_isinf_impl(context, builder, b_f_sig, [x.real])
    imag_isinf = np_real_isinf_impl(context, builder, b_f_sig, [x.imag])

    return builder.or_(real_isinf, imag_isinf)


def np_real_signbit_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)

    dispatch_table = {
        types.float32: 'numba.npymath.signbitf',
        types.float64: 'numba.npymath.signbit',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'signbit')


def np_real_copysign_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    return mathimpl.copysign_float_impl(context, builder, sig, args)

def np_real_nextafter_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.nextafterf',
        types.float64: 'numba.npymath.nextafter',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'nextafter')

def np_real_spacing_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.spacingf',
        types.float64: 'numba.npymath.spacing',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'spacing')


def np_real_ldexp_impl(context, builder, sig, args):
    # this one is slightly different to other ufuncs.
    # arguments are not homogeneous and second arg may come as
    # an 'i' or an 'l'.

    dispatch_table = {
        types.float32: 'numba.npymath.ldexpf',
        types.float64: 'numba.npymath.ldexp',
    }

    # the functions expect the second argument to be have a C int type
    x1, x2 = args
    ty1, ty2 = sig.args
    # note that types.intc should be equivalent to int_ that is
    # 'NumPy's default int')
    x2 = context.cast(builder, x2, ty2, types.intc)
    f_fi_sig = typing.signature(ty1, ty1, types.intc)
    return _dispatch_func_by_name_type(context, builder, f_fi_sig, [x1, x2],
                                       dispatch_table, 'ldexp')
