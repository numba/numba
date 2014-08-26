"""Codegen for functions used as kernels in NumPy functions

Typically, the kernels of several ufuncs that can't map directly to
Python builtins
"""

from __future__ import print_function, absolute_import, division

import math

from llvm import core as lc

from .. import cgutils, typing, types, lowering

# some NumPy constants. Note that we could generate some of them using
# the math library, but having the values copied from npy_math seems to
# yield more accurate results
_NPY_LOG2E  = 1.442695040888963407359924681001892137 # math.log(math.e, 2)
_NPY_LOG10E = 0.434294481903251827651128918916605082 # math.log(math.e, 10)
_NPY_LOGE2  = 0.693147180559945309417232121458176568 # math.log(2)


def _check_arity_and_homogeneous(sig, args, arity):
    """checks that the following are true:
    - args and sig.args have arg_count elements
    - all types are homogeneous
    """
    assert len(args) == arity
    assert len(sig.args) == arity
    ty = sig.args[0]
    # must have homogeneous args
    assert all(arg==ty for arg in sig.args) and sig.return_type == ty


def _call_func_by_name_with_cast(context, builder, sig, args,
                                 func_name, ty=types.float64):
    # it is quite common in NumPy to have loops implemented as a call
    # to the double version of the function, wrapped in casts. This
    # helper function facilitates that.
    mod = cgutils.get_module(builder)
    lty = context.get_argument_type(ty)
    fnty = lc.Type.function(lty, [lty]*len(sig.args))
    fn = mod.get_or_insert_function(fnty, name=func_name)
    cast_args = [context.cast(builder, arg, argty, ty)
             for arg, argty in zip(args, sig.args) ]

    result = builder.call(fn, cast_args)
    return context.cast(builder, result, types.float64, sig.return_type)


def _dispatch_func_by_name_type(context, builder, sig, args, table, user_name):
    # assumes types in the sig are homogeneous.
    # assumes that the function pointed by func_name has the type
    # signature sig (but needs translation to llvm types).

    ty = sig.return_type
    try:
        func_name = table[ty] # any would do... homogeneous
    except KeyError as e:
        raise LoweringError("No {0} function for real type {1}".format(user_name, str(e)))

    mod = cgutils.get_module(builder)
    if ty in types.complex_domain:
        # In numba struct types are always passed by pointer. So the call has to
        # be transformed from "result = func(ops...)" to "func(&result, ops...).
        # note that the result value pointer as first argument is the convention
        # used by numba.

        # First, prepare the return value
        complex_class = context.make_complex(ty)
        out = complex_class(context, builder)
        call_args = [out._getvalue()] + list(args)
        # get_value_as_argument for struct types like complex allocate stack space
        # and initialize with the value, the return value is the pointer to that
        # allocated space (ie: pointer to a copy of the value in the stack).
        # get_argument_type returns a pointer to the struct type in consonance.
        call_argtys = [ty] + list(sig.args)
        call_argltys = [context.get_argument_type(ty) for ty in call_argtys]
        fnty = lc.Type.function(lc.Type.void(), call_argltys)
        fn = mod.get_or_insert_function(fnty, name=func_name)

        call_args = [context.get_value_as_argument(builder, argty, arg)
                     for argty, arg in zip(call_argtys, call_args)]
        builder.call(fn, call_args)
        retval = builder.load(call_args[0])
    else:
        argtypes = [context.get_argument_type(aty) for aty in sig.args]
        restype = context.get_argument_type(sig.return_type)
        fnty = lc.Type.function(restype, argtypes)
        fn = mod.get_or_insert_function(fnty, name=func_name)
        retval = context.call_external_function(builder, fn, sig.args, args)
    return retval



def np_dummy_return_arg(context, builder, sig, args):
    # sometimes a loop does nothing other than returning the first arg...
    # for example, conjugate for non-complex numbers
    # this function implements this.
    _check_arity_and_homogeneous(sig, args, 1)
    return args[0] # nothing to do...



########################################################################
# Division kernels inspired by NumPy loops.c.src code
#
# The builtins are not applicable as they rely on a test for zero in the
# denominator. If it is zero the appropriate exception is raised.
# In NumPy, a division by zero does not raise an exception, but instead
# generated a known value. Note that a division by zero in any of the
# operations of a vector may raise an exception or issue a warning
# depending on the numpy.seterr configuration. This is not supported
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
    # in NumPy real div has the same semantics as an fdiv for generating
    # NANs, INF and NINF
    num, den = args
    lltype = num.type
    assert all(i.type==lltype for i in args), "must have homogeneous types"
    return builder.fdiv(*args)


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

    complexClass = context.make_complex(sig.args[0])
    in1, in2 = [complexClass(context, builder, value=arg) for arg in args]

    in1r = in1.real  # numerator.real
    in1i = in1.imag  # numerator.imag
    in2r = in2.real  # denominator.real
    in2i = in2.imag  # denominator.imag
    ftype = in1r.type
    assert all([i.type==ftype for i in [in1r, in1i, in2r, in2i]]), "mismatched types"
    out = complexClass(context, builder)

    ZERO = lc.Constant.real(ftype, 0.0)
    ONE = lc.Constant.real(ftype, 1.0)

    # if abs(denominator.real) >= abs(denominator.imag)
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp(lc.FCMP_OGE, in2r_abs, in2i_abs)
    with cgutils.ifelse(builder, in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            # if abs(denominator.real) == 0 and abs(denominator.imag) == 0
            in2r_is_zero = builder.fcmp(lc.FCMP_OEQ, in2r_abs, ZERO)
            in2i_is_zero = builder.fcmp(lc.FCMP_OEQ, in2i_abs, ZERO)
            in2_is_zero = builder.and_(in2r_is_zero, in2i_is_zero)
            with cgutils.ifelse(builder, in2_is_zero) as (inn_then, inn_otherwise):
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

    complexClass = context.make_complex(sig.args[0])
    in1, in2 = [complexClass(context, builder, value=arg) for arg in args]

    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type==ftype for i in [in1r, in1i, in2r, in2i]]), "mismatched types"

    ZERO = lc.Constant.real(ftype, 0.0)

    out = complexClass(context, builder)
    out.imag = ZERO

    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp(lc.FCMP_OGE, in2r_abs, in2i_abs)

    with cgutils.ifelse(builder, in2r_abs_ge_in2i_abs) as (then, otherwise):
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

def np_int_power_impl(context, builder, sig, args):
    # In NumPy ufunc loops, integer power is performed using the double
    # version of power with the appropriate casts
    assert len(args) == 2
    assert len(sig.args) == 2
    ty = sig.args[0]
    # must have homogeneous args
    assert all(arg==ty for arg in sig.args) and sig.return_type == ty

    return _call_func_by_name_with_cast(context, builder, sig, args,
                                        'numba.npymath.pow', types.float64)


def np_real_power_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 2)

    dispatch_table = {
        types.float32: 'numba.npymath.powf',
        types.float64: 'numba.npymath.pow',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'power')


def np_complex_power_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 2)

    dispatch_table = {
        types.complex64: 'numba.npymath.cpowf',
        types.complex128: 'numba.npymath.cpow',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'power')


def np_real_floor_impl(context, builder, sig, args):
    assert len(args) == 1
    assert len(sig.args) == 1
    ty = sig.args[0]
    assert ty == sig.return_type, "must have homogeneous types"
    mod = cgutils.get_module(builder)
    if ty == types.float64:
        fnty = lc.Type.function(lc.Type.double(), [lc.Type.double()])
        fn = mod.get_or_insert_function(fnty, name="numba.npymath.floor")
    elif ty == types.float32:
        fnty = lc.Type.function(lc.Type.float(), [lc.Type.float()])
        fn = mod.get_or_insert_function(fnty, name="numba.npymath.floorf")
    else:
        raise LoweringError("No floor function for real type {0}".format(str(ty)))

    return builder.call(fn, args)


########################################################################
# Numpy style complex sign

def np_complex_sign_impl(context, builder, sig, args):
    # equivalent to complex sign in NumPy's sign
    # but implemented via selects, balancing the 4 cases.
    _check_arity_and_homogeneous(sig, args, 1)
    op = args[0]
    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)

    ZERO = context.get_constant(float_ty, 0.0)
    ONE  = context.get_constant(float_ty, 1.0)
    MINUS_ONE = context.get_constant(float_ty, -1.0)
    NAN = context.get_constant(float_ty, float('nan'))
    result = complex_class(context, builder)
    result.real = ZERO
    result.imag = ZERO

    cmp_sig = typing.signature(*[ty] * 3)
    cmp_args = [op, result._getvalue()]
    arg1_ge_arg2 = np_complex_greater_equal_impl(context, builder, cmp_sig, cmp_args)
    arg1_eq_arg2 = np_complex_equal_impl(context, builder, cmp_sig, cmp_args)
    arg1_lt_arg2 = np_complex_lower_impl(context, builder, cmp_sig, cmp_args)

    real_when_ge = builder.select(arg1_eq_arg2, ZERO, ONE)
    real_when_nge = builder.select(arg1_lt_arg2, MINUS_ONE, NAN)
    result.real = builder.select(arg1_ge_arg2, real_when_ge, real_when_nge)

    return result._getvalue()


########################################################################
# Numpy rint

def np_real_rint_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.rintf',
        types.float64: 'numba.npymath.rint',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'rint')


def np_complex_rint_impl(context, builder, sig, args):
    # based on code in NumPy's funcs.inc.src
    # rint of a complex number defined as rint of its real and imag
    # parts
    _check_arity_and_homogeneous(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)
    in1 = complex_class(context, builder, value=args[0])
    out = complex_class(context, builder)

    inner_sig = typing.signature(*[float_ty]*2)
    out.real = np_real_rint_impl(context, builder, inner_sig, [in1.real])
    out.imag = np_real_rint_impl(context, builder, inner_sig, [in1.imag])
    return out._getvalue()


########################################################################
# NumPy conj/conjugate
def np_complex_conjugate_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)
    in1 = complex_class(context, builder, value=args[0])
    out = complex_class(context, builder)
    ZERO = context.get_constant(float_ty, 0.0)
    out.real = in1.real
    out.imag = builder.fsub(ZERO, in1.imag)
    return out._getvalue()


########################################################################
# NumPy exp

def np_real_exp_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.expf',
        types.float64: 'numba.npymath.exp',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp')


def np_complex_exp_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.cexpf',
        types.complex128: 'numba.npymath.cexp',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp')

########################################################################
# NumPy exp2

def np_real_exp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.exp2f',
        types.float64: 'numba.npymath.exp2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'exp2')


def np_complex_exp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)
    in1 = complex_class(context, builder, value=args[0])
    tmp = complex_class(context, builder)
    loge2 = context.get_constant(float_ty, _NPY_LOGE2)
    tmp.real = builder.fmul(loge2, in1.real)
    tmp.imag = builder.fmul(loge2, in1.imag)
    return np_complex_exp_impl(context, builder, sig, [tmp._getvalue()])


########################################################################
# NumPy log

def np_real_log_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.logf',
        types.float64: 'numba.npymath.log',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log')


def np_complex_log_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.complex64: 'numba.npymath.clogf',
        types.complex128: 'numba.npymath.clog',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log')

########################################################################
# NumPy log2

def np_real_log2_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.log2f',
        types.float64: 'numba.npymath.log2',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log2')

def np_complex_log2_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = complex_class(context, builder, value=tmp)
    log2e = context.get_constant(float_ty, _NPY_LOG2E)
    tmp.real = builder.fmul(log2e, tmp.real)
    tmp.imag = builder.fmul(log2e, tmp.imag)
    return tmp._getvalue()


########################################################################
# NumPy log10

def np_real_log10_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.log10f',
        types.float64: 'numba.npymath.log10',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'log10')


def np_complex_log10_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    complex_class = context.make_complex(ty)
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = complex_class(context, builder, value=tmp)
    log10e = context.get_constant(float_ty, _NPY_LOG10E)
    tmp.real = builder.fmul(log10e, tmp.real)
    tmp.imag = builder.fmul(log10e, tmp.imag)
    return tmp._getvalue()


########################################################################
# NumPy expm1

def np_real_expm1_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.expm1f',
        types.float64: 'numba.npymath.expm1',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'expm1')

def np_complex_expm1_impl(context, builder, sig, args):
    # this is based on nc_expm1 in funcs.inc.src
    _check_arity_and_homogeneous(sig, args, 1)

    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty]*2)
    complex_class = context.make_complex(ty)

    MINUS_ONE = context.get_constant(float_ty, -1.0)
    in1 = complex_class(context, builder, value=args[0])
    a = np_real_exp_impl(context, builder, float_unary_sig, [in1.real])
    out = complex_class(context, builder)
    cos_imag = np_real_cos_impl(context, builder, float_unary_sig, [in1.imag])
    sin_imag = np_real_sin_impl(context, builder, float_unary_sig, [in1.imag])
    tmp = builder.fmul(a, cos_imag)
    out.imag = builder.fmul(a, sin_imag)
    out.real = builder.fadd(tmp, MINUS_ONE)

    return out._getvalue()


########################################################################
# NumPy sin

def np_real_sin_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.sinf',
        types.float64: 'numba.npymath.sin',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'sin')


########################################################################
# NumPy cos

def np_real_cos_impl(context, builder, sig, args):
    _check_arity_and_homogeneous(sig, args, 1)

    dispatch_table = {
        types.float32: 'numba.npymath.cosf',
        types.float64: 'numba.npymath.cos',
    }

    return _dispatch_func_by_name_type(context, builder, sig, args,
                                       dispatch_table, 'cos')


########################################################################
# NumPy style complex predicates

def np_complex_greater_equal_impl(context, builder, sig, args):
    # equivalent to macro CGE in NumPy's loops.c.src
    # ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi >= yi))
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
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


def np_complex_lower_equal_impl(context, builder, sig, args):
    # equivalent to macro CLE in NumPy's loops.c.src
    # ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi <= yi))
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
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


def np_complex_greater_impl(context, builder, sig, args):
    # equivalent to macro CGT in NumPy's loops.c.src
    # ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi > yi))
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
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


def np_complex_lower_impl(context, builder, sig, args):
    # equivalent to macro CLT in NumPy's loops.c.src
    # ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) || (xr == yr && xi < yi))
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
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


def np_complex_equal_impl(context, builder, sig, args):
    # equivalent to macro CEQ in NumPy's loops.c.src
    # (xr == yr && xi == yi)
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_eq_yr = builder.fcmp(lc.FCMP_OEQ, xr, yr)
    xi_eq_yi = builder.fcmp(lc.FCMP_OEQ, xi, yi)
    return builder.and_(xr_eq_yr, xi_eq_yi)


def np_complex_not_equal_impl(complex, builder, sig, args):
    # equivalent to marcro CNE in NumPy's loops.c.src
    # (xr != yr || xi != yi)
    _check_arity_and_homogeneous(sig, args, 2)

    complex_class = context.make_complex(sig.args[0])
    in1, in2 = [complex_class(context, builder, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag

    xr_ne_yr = builder.fcmp(lc.FCMP_ONE, xr, yr)
    xi_ne_yi = builder.fcmp(lc.FCMP_ONE, xi, yi)
    return builder.or_(xr_ne_yr, xi_ne_yi)
