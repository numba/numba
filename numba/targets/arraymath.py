"""
Implementation of math operations on Array objects.
"""

from __future__ import print_function, absolute_import, division

import math

import numpy as np

from llvmlite import ir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Constant, Type

from numba import jit, types, cgutils, typing
from numba.extending import overload, overload_method
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (lower_builtin, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.typing import signature
from .arrayobj import make_array, load_item, store_item, _empty_nd_impl


#----------------------------------------------------------------------------
# Basic stats and aggregates

@lower_builtin(np.sum, types.Array)
@lower_builtin("array.sum", types.Array)
def array_sum(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_sum_impl(arr):
        c = zero
        for v in np.nditer(arr):
            c += v.item()
        return c

    res = context.compile_internal(builder, array_sum_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(np.prod, types.Array)
@lower_builtin("array.prod", types.Array)
def array_prod(context, builder, sig, args):

    def array_prod_impl(arr):
        c = 1
        for v in np.nditer(arr):
            c *= v.item()
        return c

    res = context.compile_internal(builder, array_prod_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(np.cumsum, types.Array)
@lower_builtin("array.cumsum", types.Array)
def array_cumsum(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)
    zero = scalar_dtype(0)

    def array_cumsum_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = np.empty(size, dtype)
        c = zero
        for idx, v in enumerate(arr.flat):
            c += v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumsum_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)



@lower_builtin(np.cumprod, types.Array)
@lower_builtin("array.cumprod", types.Array)
def array_cumprod(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)

    def array_cumprod_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = np.empty(size, dtype)
        c = 1
        for idx, v in enumerate(arr.flat):
            c *= v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumprod_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@lower_builtin(np.mean, types.Array)
@lower_builtin("array.mean", types.Array)
def array_mean(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_mean_impl(arr):
        # Can't use the naive `arr.sum() / arr.size`, as it would return
        # a wrong result on integer sum overflow.
        c = zero
        for v in np.nditer(arr):
            c += v.item()
        return c / arr.size

    res = context.compile_internal(builder, array_mean_impl, sig, args,
                                   locals=dict(c=sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.var, types.Array)
@lower_builtin("array.var", types.Array)
def array_var(context, builder, sig, args):
    def array_var_impl(arr):
        # Compute the mean
        m = arr.mean()

        # Compute the sum of square diffs
        ssd = 0
        for v in np.nditer(arr):
            ssd += (v.item() - m) ** 2
        return ssd / arr.size

    res = context.compile_internal(builder, array_var_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.std, types.Array)
@lower_builtin("array.std", types.Array)
def array_std(context, builder, sig, args):
    def array_std_impl(arry):
        return arry.var() ** 0.5
    res = context.compile_internal(builder, array_std_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.min, types.Array)
@lower_builtin("array.min", types.Array)
def array_min(context, builder, sig, args):
    ty = sig.args[0].dtype
    if isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
        # NaT is smaller than every other value, but it is
        # ignored as far as min() is concerned.
        nat = ty('NaT')

        def array_min_impl(arry):
            min_value = nat
            it = np.nditer(arry)
            for view in it:
                v = view.item()
                if v != nat:
                    min_value = v
                    break

            for view in it:
                v = view.item()
                if v != nat and v < min_value:
                    min_value = v
            return min_value

    else:
        def array_min_impl(arry):
            it = np.nditer(arry)
            for view in it:
                min_value = view.item()
                break

            for view in it:
                v = view.item()
                if v < min_value:
                    min_value = v
            return min_value
    res = context.compile_internal(builder, array_min_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(np.max, types.Array)
@lower_builtin("array.max", types.Array)
def array_max(context, builder, sig, args):
    def array_max_impl(arry):
        it = np.nditer(arry)
        for view in it:
            max_value = view.item()
            break

        for view in it:
            v = view.item()
            if v > max_value:
                max_value = v
        return max_value
    res = context.compile_internal(builder, array_max_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(np.argmin, types.Array)
@lower_builtin("array.argmin", types.Array)
def array_argmin(context, builder, sig, args):
    ty = sig.args[0].dtype
    # NOTE: Under Numpy < 1.10, argmin() is inconsistent with min() on NaT values:
    # https://github.com/numpy/numpy/issues/6030

    if (numpy_version >= (1, 10) and
        isinstance(ty, (types.NPDatetime, types.NPTimedelta))):
        # NaT is smaller than every other value, but it is
        # ignored as far as argmin() is concerned.
        nat = ty('NaT')

        def array_argmin_impl(arry):
            min_value = nat
            min_idx = 0
            it = arry.flat
            idx = 0
            for v in it:
                if v != nat:
                    min_value = v
                    min_idx = idx
                    idx += 1
                    break
                idx += 1

            for v in it:
                if v != nat and v < min_value:
                    min_value = v
                    min_idx = idx
                idx += 1
            return min_idx

    else:
        def array_argmin_impl(arry):
            for v in arry.flat:
                min_value = v
                min_idx = 0
                break

            idx = 0
            for v in arry.flat:
                if v < min_value:
                    min_value = v
                    min_idx = idx
                idx += 1
            return min_idx
    res = context.compile_internal(builder, array_argmin_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.argmax, types.Array)
@lower_builtin("array.argmax", types.Array)
def array_argmax(context, builder, sig, args):
    def array_argmax_impl(arry):
        for v in arry.flat:
            max_value = v
            max_idx = 0
            break

        idx = 0
        for v in arry.flat:
            if v > max_value:
                max_value = v
                max_idx = idx
            idx += 1
        return max_idx
    res = context.compile_internal(builder, array_argmax_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@overload(np.all)
@overload_method(types.Array, "all")
def np_all(a):
    def flat_all(a):
        for v in np.nditer(a):
            if not v.item():
                return False
        return True

    return flat_all

@overload(np.any)
@overload_method(types.Array, "any")
def np_any(a):
    def flat_any(a):
        for v in np.nditer(a):
            if v.item():
                return True
        return False

    return flat_any


def get_isnan(dtype):
    """
    A generic isnan() function
    """
    if isinstance(dtype, (types.Float, types.Complex)):
        return np.isnan
    else:
        @jit(nopython=True)
        def _trivial_isnan(x):
            return False
        return _trivial_isnan


@overload(np.nanmin)
def np_nanmin(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanmin_impl(a):
        if a.size == 0:
            raise ValueError("nanmin(): empty array")
        for view in np.nditer(a):
            minval = view.item()
            break
        for view in np.nditer(a):
            v = view.item()
            if not minval < v and not isnan(v):
                minval = v
        return minval

    return nanmin_impl

@overload(np.nanmax)
def np_nanmax(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanmax_impl(a):
        if a.size == 0:
            raise ValueError("nanmin(): empty array")
        for view in np.nditer(a):
            maxval = view.item()
            break
        for view in np.nditer(a):
            v = view.item()
            if not maxval > v and not isnan(v):
                maxval = v
        return maxval

    return nanmax_impl

if numpy_version >= (1, 8):
    @overload(np.nanmean)
    def np_nanmean(a):
        if not isinstance(a, types.Array):
            return
        isnan = get_isnan(a.dtype)

        def nanmean_impl(arr):
            c = 0.0
            count = 0
            for view in np.nditer(arr):
                v = view.item()
                if not isnan(v):
                    c += v.item()
                    count += 1
            # np.divide() doesn't raise ZeroDivisionError
            return np.divide(c, count)

        return nanmean_impl

    @overload(np.nanvar)
    def np_nanvar(a):
        if not isinstance(a, types.Array):
            return
        isnan = get_isnan(a.dtype)

        def nanvar_impl(arr):
            # Compute the mean
            m = np.nanmean(arr)

            # Compute the sum of square diffs
            ssd = 0.0
            count = 0
            for view in np.nditer(arr):
                v = view.item()
                if not isnan(v):
                    ssd += (v.item() - m) ** 2
                    count += 1
            # np.divide() doesn't raise ZeroDivisionError
            return np.divide(ssd, count)

        return nanvar_impl

    @overload(np.nanstd)
    def np_nanstd(a):
        if not isinstance(a, types.Array):
            return

        def nanstd_impl(arr):
            return np.nanvar(arr) ** 0.5

        return nanstd_impl

@overload(np.nansum)
def np_nansum(a):
    if not isinstance(a, types.Array):
        return
    if isinstance(a.dtype, types.Integer):
        retty = types.intp
    else:
        retty = a.dtype
    zero = retty(0)
    isnan = get_isnan(a.dtype)

    def nansum_impl(arr):
        c = zero
        for view in np.nditer(arr):
            v = view.item()
            if not isnan(v):
                c += v
        return c

    return nansum_impl


#----------------------------------------------------------------------------
# Median and partitioning

@jit(nopython=True)
def _partition(A, low, high):
    mid = (low + high) >> 1
    # NOTE: the pattern of swaps below for the pivot choice and the
    # partitioning gives good results (i.e. regular O(n log n))
    # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
    # risk breaking this property.

    # Use median of three {low, middle, high} as the pivot
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
    if A[high] < A[mid]:
        A[high], A[mid] = A[mid], A[high]
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
    pivot = A[mid]

    A[high], A[mid] = A[mid], A[high]

    i = low
    for j in range(low, high):
        if A[j] <= pivot:
            A[i], A[j] = A[j], A[i]
            i += 1

    A[i], A[high] = A[high], A[i]
    return i

@jit(nopython=True)
def _select(arry, k, low, high):
    """
    Select the k'th smallest element in array[low:high + 1].
    """
    i = _partition(arry, low, high)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(arry, low, high)
        else:
            high = i - 1
            i = _partition(arry, low, high)
    return arry[k]

@jit(nopython=True)
def _select_two(arry, k, low, high):
    """
    Select the k'th and k+1'th smallest elements in array[low:high + 1].

    This is significantly faster than doing two independent selections
    for k and k+1.
    """
    while True:
        assert high > low  # by construction
        i = _partition(arry, low, high)
        if i < k:
            low = i + 1
        elif i > k + 1:
            high = i - 1
        elif i == k:
            _select(arry, k + 1, i + 1, high)
            break
        else:  # i == k + 1
            _select(arry, k, low, i - 1)
            break

    return arry[k], arry[k + 1]

@jit(nopython=True)
def _median_inner(temp_arry, n):
    """
    The main logic of the median() call.  *temp_arry* must be disposable,
    as this function will mutate it.
    """
    low = 0
    high = n - 1
    half = n >> 1
    if n & 1 == 0:
        a, b = _select_two(temp_arry, half - 1, low, high)
        return (a + b) / 2
    else:
        return _select(temp_arry, half, low, high)

@overload(np.median)
def np_median(a):
    if not isinstance(a, types.Array):
        return

    def median_impl(arry):
        # np.median() works on the flattened array, and we need a temporary
        # workspace anyway
        temp_arry = arry.flatten()
        n = temp_arry.shape[0]
        return _median_inner(temp_arry, n)

    return median_impl

if numpy_version >= (1, 9):
    @overload(np.nanmedian)
    def np_nanmedian(a):
        if not isinstance(a, types.Array):
            return
        isnan = get_isnan(a.dtype)

        def nanmedian_impl(arry):
            # Create a temporary workspace with only non-NaN values
            temp_arry = np.empty(arry.size, arry.dtype)
            n = 0
            for view in np.nditer(arry):
                v = view.item()
                if not isnan(v):
                    temp_arry[n] = v
                    n += 1
            return _median_inner(temp_arry, n)

        return nanmedian_impl


#----------------------------------------------------------------------------
# Element-wise computations

def _np_round_intrinsic(tp):
    # np.round() always rounds half to even
    return "llvm.rint.f%d" % (tp.bitwidth,)

def _np_round_float(context, builder, tp, val):
    llty = context.get_value_type(tp)
    module = builder.module
    fnty = lc.Type.function(llty, [llty])
    fn = module.get_or_insert_function(fnty, name=_np_round_intrinsic(tp))
    return builder.call(fn, (val,))

@lower_builtin(np.round, types.Float)
def scalar_round_unary(context, builder, sig, args):
    res =  _np_round_float(context, builder, sig.args[0], args[0])
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.round, types.Integer)
def scalar_round_unary(context, builder, sig, args):
    res = args[0]
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.round, types.Complex)
def scalar_round_unary_complex(context, builder, sig, args):
    fltty = sig.args[0].underlying_float
    z = context.make_complex(builder, sig.args[0], args[0])
    z.real = _np_round_float(context, builder, fltty, z.real)
    z.imag = _np_round_float(context, builder, fltty, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.round, types.Float, types.Integer)
@lower_builtin(np.round, types.Integer, types.Integer)
def scalar_round_binary_float(context, builder, sig, args):
    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x

        # NOTE: this is CPython's algorithm, but perhaps this is overkill
        # when emulating Numpy's behaviour.
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
            return (np.round(y) / pow2) / pow1

        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return np.round(y) * pow1

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.round, types.Complex, types.Integer)
def scalar_round_binary_complex(context, builder, sig, args):
    def round_ndigits(z, ndigits):
        return complex(np.round(z.real, ndigits),
                       np.round(z.imag, ndigits))

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.round, types.Array, types.Integer,
           types.Array)
def array_round(context, builder, sig, args):
    def array_round_impl(arr, decimals, out):
        if arr.shape != out.shape:
            raise ValueError("invalid output shape")
        for index, val in np.ndenumerate(arr):
            out[index] = np.round(val, decimals)
        return out

    res = context.compile_internal(builder, array_round_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@lower_builtin(np.sinc, types.Array)
def array_sinc(context, builder, sig, args):
    def array_sinc_impl(arr):
        out = np.zeros_like(arr)
        for index, val in np.ndenumerate(arr):
            out[index] = np.sinc(val)
        return out
    res = context.compile_internal(builder, array_sinc_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@lower_builtin(np.sinc, types.Number)
def scalar_sinc(context, builder, sig, args):
    scalar_dtype = sig.return_type
    def scalar_sinc_impl(val):
        if val == 0.e0: # to match np impl
            val = 1e-20
        val *= np.pi # np sinc is the normalised variant
        return np.sin(val)/val
    res = context.compile_internal(builder, scalar_sinc_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.angle, types.Number)
@lower_builtin(np.angle, types.Number, types.Boolean)
def scalar_angle_kwarg(context, builder, sig, args):
    deg_mult = sig.return_type(180 / np.pi)
    def scalar_angle_impl(val, deg):
        if deg:
            return np.arctan2(val.imag, val.real) * deg_mult
        else:
            return np.arctan2(val.imag, val.real)

    if len(args) == 1:
        args = args + (cgutils.false_bit,)
        sig = signature(sig.return_type, *(sig.args + (types.boolean,)))
    res = context.compile_internal(builder, scalar_angle_impl,
                                   sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(np.angle, types.Array)
@lower_builtin(np.angle, types.Array, types.Boolean)
def array_angle_kwarg(context, builder, sig, args):
    arg = sig.args[0]
    ret_dtype = sig.return_type.dtype

    def array_angle_impl(arr, deg):
        out = np.zeros_like(arr, dtype=ret_dtype)
        for index, val in np.ndenumerate(arr):
            out[index] = np.angle(val, deg)
        return out

    if len(args) == 1:
        args = args + (cgutils.false_bit,)
        sig = signature(sig.return_type, *(sig.args + (types.boolean,)))

    res = context.compile_internal(builder, array_angle_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@lower_builtin(np.nonzero, types.Array)
@lower_builtin("array.nonzero", types.Array)
@lower_builtin(np.where, types.Array)
def array_nonzero(context, builder, sig, args):
    aryty = sig.args[0]
    # Return type is a N-tuple of 1D C-contiguous arrays
    retty = sig.return_type
    outaryty = retty.dtype
    ndim = aryty.ndim
    nouts = retty.count

    ary = make_array(aryty)(context, builder, args[0])
    shape = cgutils.unpack_tuple(builder, ary.shape)
    strides = cgutils.unpack_tuple(builder, ary.strides)
    data = ary.data
    layout = aryty.layout

    # First count the number of non-zero elements
    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)
    count = cgutils.alloca_once_value(builder, zero)
    with cgutils.loop_nest(builder, shape, zero.type) as indices:
        ptr = cgutils.get_item_pointer2(builder, data, shape, strides,
                                        layout, indices)
        val = load_item(context, builder, aryty, ptr)
        nz = context.is_true(builder, aryty.dtype, val)
        with builder.if_then(nz):
            builder.store(builder.add(builder.load(count), one), count)

    # Then allocate output arrays of the right size
    out_shape = (builder.load(count),)
    outs = [_empty_nd_impl(context, builder, outaryty, out_shape)._getvalue()
            for i in range(nouts)]
    outarys = [make_array(outaryty)(context, builder, out) for out in outs]
    out_datas = [out.data for out in outarys]

    # And fill them up
    index = cgutils.alloca_once_value(builder, zero)
    with cgutils.loop_nest(builder, shape, zero.type) as indices:
        ptr = cgutils.get_item_pointer2(builder, data, shape, strides,
                                        layout, indices)
        val = load_item(context, builder, aryty, ptr)
        nz = context.is_true(builder, aryty.dtype, val)
        with builder.if_then(nz):
            # Store element indices in output arrays
            if not indices:
                # For a 0-d array, store 0 in the unique output array
                indices = (zero,)
            cur = builder.load(index)
            for i in range(nouts):
                ptr = cgutils.get_item_pointer2(builder, out_datas[i],
                                                out_shape, (),
                                                'C', [cur])
                store_item(context, builder, outaryty, indices[i], ptr)
            builder.store(builder.add(cur, one), index)

    tup = context.make_tuple(builder, sig.return_type, outs)
    return impl_ret_new_ref(context, builder, sig.return_type, tup)


def array_where(context, builder, sig, args):
    """
    np.where(array, array, array)
    """
    layouts = set(a.layout for a in sig.args)
    if layouts == set('C'):
        # Faster implementation for C-contiguous arrays
        def where_impl(cond, x, y):
            shape = cond.shape
            if x.shape != shape or y.shape != shape:
                raise ValueError("all inputs should have the same shape")
            res = np.empty_like(x)
            cf = cond.flat
            xf = x.flat
            yf = y.flat
            rf = res.flat
            for i in range(cond.size):
                rf[i] = xf[i] if cf[i] else yf[i]
            return res
    else:

        def where_impl(cond, x, y):
            shape = cond.shape
            if x.shape != shape or y.shape != shape:
                raise ValueError("all inputs should have the same shape")
            res = np.empty_like(x)
            for idx, c in np.ndenumerate(cond):
                res[idx] = x[idx] if c else y[idx]
            return res

    res = context.compile_internal(builder, where_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(np.where, types.Any, types.Any, types.Any)
def any_where(context, builder, sig, args):
    cond = sig.args[0]
    if isinstance(cond, types.Array):
        return array_where(context, builder, sig, args)

    def scalar_where_impl(cond, x, y):
        """
        np.where(scalar, scalar, scalar): return a 0-dim array
        """
        scal = x if cond else y
        # This is the equivalent of np.full_like(scal, scal),
        # for compatibility with Numpy < 1.8
        arr = np.empty_like(scal)
        arr[()] = scal
        return arr

    res = context.compile_internal(builder, scalar_where_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


#----------------------------------------------------------------------------
# Misc functions

@overload(np.diff)
def np_diff_impl(a, n=1):
    if not isinstance(a, types.Array) or a.ndim == 0:
        return

    def diff_impl(a, n=1):
        if n == 0:
            return a.copy()
        if n < 0:
            raise ValueError("diff(): order must be non-negative")
        size = a.shape[-1]
        out_shape = a.shape[:-1] + (max(size - n, 0),)
        out = np.empty(out_shape, a.dtype)
        if out.size == 0:
            return out

        # np.diff() works on each last dimension subarray independently.
        # To make things easier, normalize input and output into 2d arrays
        a2 = a.reshape((-1, size))
        out2 = out.reshape((-1, out.shape[-1]))
        # A scratchpad for subarrays
        work = np.empty(size, a.dtype)

        for major in range(a2.shape[0]):
            # First iteration: diff a2 into work
            for i in range(size - 1):
                work[i] = a2[major, i + 1] - a2[major, i]
            # Other iterations: diff work into itself
            for niter in range(1, n):
                for i in range(size - niter - 1):
                    work[i] = work[i + 1] - work[i]
            # Copy final diff into out2
            out2[major] = work[:size - n]

        return out

    return diff_impl


def validate_1d_array_like(func_name, seq):
    if isinstance(seq, types.Array):
        if seq.ndim != 1:
            raise TypeError("{0}(): input should have dimension 1"
                            .format(func_name))
    elif not isinstance(seq, types.Sequence):
        raise TypeError("{0}(): input should be an array or sequence"
                        .format(func_name))


@overload(np.bincount)
def np_bincount(a, weights=None):
    validate_1d_array_like("bincount", a)
    if not isinstance(a.dtype, types.Integer):
        return

    if weights not in (None, types.none):
        validate_1d_array_like("bincount", weights)
        out_dtype = weights.dtype

        @jit(nopython=True)
        def validate_inputs(a, weights):
            if len(a) != len(weights):
                raise ValueError("bincount(): weights and list don't have the same length")

        @jit(nopython=True)
        def count_item(out, idx, val, weights):
            out[val] += weights[idx]

    else:
        out_dtype = types.intp

        @jit(nopython=True)
        def validate_inputs(a, weights):
            pass

        @jit(nopython=True)
        def count_item(out, idx, val, weights):
            out[val] += 1

    def bincount_impl(a, weights=None):
        validate_inputs(a, weights)
        n = len(a)

        a_max = a[0] if n > 0 else -1
        for i in range(1, n):
            if a[i] < 0:
                raise ValueError("bincount(): first argument must be non-negative")
            a_max = max(a_max, a[i])

        out = np.zeros(a_max + 1, out_dtype)
        for i in range(n):
            count_item(out, i, a[i], weights)
        return out

    return bincount_impl


@overload(np.searchsorted)
def searchsorted(a, v):
    if isinstance(v, types.Array):
        # N-d array and output

        def searchsorted_impl(a, v):
            out = np.empty(v.shape, np.intp)
            for view, outview in np.nditer((v, out)):
                index = np.searchsorted(a, view.item())
                outview.itemset(index)
            return out

        return searchsorted_impl

    elif isinstance(v, types.Sequence):
        # 1-d sequence and output

        def searchsorted_impl(a, v):
            out = np.empty(len(v), np.intp)
            for i in range(len(v)):
                out[i] = np.searchsorted(a, v[i])
            return out

        return searchsorted_impl

    else:
        # Scalar value and output
        # Note: NaNs come last in Numpy-sorted arrays

        def searchsorted_impl(a, v):
            n = len(a)
            if np.isnan(v):
                # Find the first nan (i.e. the last from the end of a,
                # since there shouldn't be many of them in practice)
                for i in range(n, 0, -1):
                    if not np.isnan(a[i - 1]):
                        return i
                return 0
            lo = 0
            hi = n
            while hi > lo:
                mid = (lo + hi) >> 1
                if a[mid] < v:
                    # mid is too low => go up
                    lo = mid + 1
                else:
                    # mid is too high, or is a NaN => go down
                    hi = mid
            return lo

        return searchsorted_impl


@overload(np.digitize)
def np_digitize(x, bins, right=False):
    @jit(nopython=True)
    def are_bins_increasing(bins):
        n = len(bins)
        is_increasing = True
        is_decreasing = True
        if n > 1:
            prev = bins[0]
            for i in range(1, n):
                cur = bins[i]
                is_increasing = is_increasing and not prev > cur
                is_decreasing = is_decreasing and not prev < cur
                if not is_increasing and not is_decreasing:
                    raise ValueError("bins must be monotonically increasing or decreasing")
                prev = cur
        return is_increasing

    # NOTE: the algorithm is slightly different from searchsorted's,
    # as the edge cases (bin boundaries, NaN) give different results.

    @jit(nopython=True)
    def digitize_scalar(x, bins, right):
        # bins are monotonically-increasing
        n = len(bins)
        lo = 0
        hi = n

        if right:
            if np.isnan(x):
                # Find the first nan (i.e. the last from the end of bins,
                # since there shouldn't be many of them in practice)
                for i in range(n, 0, -1):
                    if not np.isnan(bins[i - 1]):
                        return i
                return 0
            while hi > lo:
                mid = (lo + hi) >> 1
                if bins[mid] < x:
                    # mid is too low => narrow to upper bins
                    lo = mid + 1
                else:
                    # mid is too high, or is a NaN => narrow to lower bins
                    hi = mid
        else:
            if np.isnan(x):
                # NaNs end up in the last bin
                return n
            while hi > lo:
                mid = (lo + hi) >> 1
                if bins[mid] <= x:
                    # mid is too low => narrow to upper bins
                    lo = mid + 1
                else:
                    # mid is too high, or is a NaN => narrow to lower bins
                    hi = mid

        return lo

    @jit(nopython=True)
    def digitize_scalar_decreasing(x, bins, right):
        # bins are monotonically-decreasing
        n = len(bins)
        lo = 0
        hi = n

        if right:
            if np.isnan(x):
                # Find the last nan
                for i in range(0, n):
                    if not np.isnan(bins[i]):
                        return i
                return n
            while hi > lo:
                mid = (lo + hi) >> 1
                if bins[mid] < x:
                    # mid is too high => narrow to lower bins
                    hi = mid
                else:
                    # mid is too low, or is a NaN => narrow to upper bins
                    lo = mid + 1
        else:
            if np.isnan(x):
                # NaNs end up in the first bin
                return 0
            while hi > lo:
                mid = (lo + hi) >> 1
                if bins[mid] <= x:
                    # mid is too high => narrow to lower bins
                    hi = mid
                else:
                    # mid is too low, or is a NaN => narrow to upper bins
                    lo = mid + 1

        return lo

    if isinstance(x, types.Array):
        # N-d array and output

        def digitize_impl(x, bins, right=False):
            is_increasing = are_bins_increasing(bins)
            out = np.empty(x.shape, np.intp)
            for view, outview in np.nditer((x, out)):
                if is_increasing:
                    index = digitize_scalar(view.item(), bins, right)
                else:
                    index = digitize_scalar_decreasing(view.item(), bins, right)
                outview.itemset(index)
            return out

        return digitize_impl

    elif isinstance(x, types.Sequence):
        # 1-d sequence and output

        def digitize_impl(x, bins, right=False):
            is_increasing = are_bins_increasing(bins)
            out = np.empty(len(x), np.intp)
            for i in range(len(x)):
                if is_increasing:
                    out[i] = digitize_scalar(x[i], bins, right)
                else:
                    out[i] = digitize_scalar_decreasing(x[i], bins, right)
            return out

        return digitize_impl


_range = range

@overload(np.histogram)
def np_histogram(a, bins=10, range=None):
    if isinstance(bins, (int, types.Integer)):
        # With a uniform distribution of bins, use a fast algorithm
        # independent of the number of bins

        if range in (None, types.none):
            inf = float('inf')
            def histogram_impl(a, bins=10, range=None):
                bin_min = inf
                bin_max = -inf
                for view in np.nditer(a):
                    v = view.item()
                    if bin_min > v:
                        bin_min = v
                    if bin_max < v:
                        bin_max = v
                return np.histogram(a, bins, (bin_min, bin_max))

        else:
            def histogram_impl(a, bins=10, range=None):
                if bins <= 0:
                    raise ValueError("histogram(): `bins` should be a positive integer")
                bin_min, bin_max = range
                if not bin_min <= bin_max:
                    raise ValueError("histogram(): max must be larger than min in range parameter")

                hist = np.zeros(bins, np.intp)
                if bin_max > bin_min:
                    bin_ratio = bins / (bin_max - bin_min)
                    for view in np.nditer(a):
                        v = view.item()
                        b = math.floor((v - bin_min) * bin_ratio)
                        if 0 <= b < bins:
                            hist[int(b)] += 1
                        elif v == bin_max:
                            hist[bins - 1] += 1

                bins_array = np.linspace(bin_min, bin_max, bins + 1)
                return hist, bins_array

    else:
        # With a custom bins array, use a bisection search

        def histogram_impl(a, bins, range=None):
            nbins = len(bins) - 1
            for i in _range(nbins):
                # Note this also catches NaNs
                if not bins[i] <= bins[i + 1]:
                    raise ValueError("histogram(): bins must increase monotonically")

            bin_min = bins[0]
            bin_max = bins[nbins]
            hist = np.zeros(nbins, np.intp)

            if nbins > 0:
                for view in np.nditer(a):
                    v = view.item()
                    if not bin_min <= v <= bin_max:
                        # Value is out of bounds, ignore (this also catches NaNs)
                        continue
                    # Bisect in bins[:-1]
                    lo = 0
                    hi = nbins - 1
                    while lo < hi:
                        # Note the `+ 1` is necessary to avoid an infinite
                        # loop where mid = lo => lo = mid
                        mid = (lo + hi + 1) >> 1
                        if v < bins[mid]:
                            hi = mid - 1
                        else:
                            lo = mid
                    hist[lo] += 1

            return hist, bins

    return histogram_impl
