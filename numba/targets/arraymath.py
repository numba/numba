"""
Implementation of math operations on Array objects.
"""

from __future__ import print_function, absolute_import, division

import math
from collections import namedtuple
from enum import IntEnum

import numpy as np

from llvmlite import ir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Constant, Type

from numba import types, cgutils, typing
from numba.extending import overload, overload_method, register_jitable
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (lower_builtin, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.typing import signature
from .arrayobj import make_array, load_item, store_item, _empty_nd_impl
from .linalg import ensure_blas

from numba.extending import intrinsic
from numba.errors import RequireConstValue, TypingError

def _check_blas():
    # Checks if a BLAS is available so e.g. dot will work
    try:
        ensure_blas()
    except ImportError:
        return False
    return True

_HAVE_BLAS = _check_blas()

@intrinsic
def _create_tuple_result_shape(tyctx, shape_list, shape_tuple):
    """
    This routine converts shape list where the axis dimension has already
    been popped to a tuple for indexing of the same size.  The original shape
    tuple is also required because it contains a length field at compile time
    whereas the shape list does not.
    """

    # The new tuple's size is one less than the original tuple since axis
    # dimension removed.
    nd = len(shape_tuple) - 1
    # The return type of this intrinsic is an int tuple of length nd.
    tupty = types.UniTuple(types.intp, nd)
    # The function signature for this intrinsic.
    function_sig = tupty(shape_list, shape_tuple)

    def codegen(cgctx, builder, signature, args):
        lltupty = cgctx.get_value_type(tupty)
        # Create an empty int tuple.
        tup = cgutils.get_null_value(lltupty)

        # Get the shape list from the args and we don't need shape tuple.
        [in_shape, _] = args

        def array_indexer(a, i):
            return a[i]

        # loop to fill the tuple
        for i in range(nd):
            dataidx = cgctx.get_constant(types.intp, i)
            # compile and call array_indexer
            data = cgctx.compile_internal(builder, array_indexer,
                                          types.intp(shape_list, types.intp),
                                          [in_shape, dataidx])
            tup = builder.insert_value(tup, data, i)
        return tup

    return function_sig, codegen

@intrinsic(support_literals=True)
def _gen_index_tuple(tyctx, shape_tuple, value, axis):
    """
    Generates a tuple that can be used to index a specific slice from an
    array for sum with axis.  shape_tuple is the size of the dimensions of
    the input array.  'value' is the value to put in the indexing tuple
    in the axis dimension and 'axis' is that dimension.  For this to work,
    axis has to be a const.
    """
    if not isinstance(axis, types.Const):
        raise RequireConstValue('axis argument must be a constant')
    # Get the value of the axis constant.
    axis_value = axis.value
    # The length of the indexing tuple to be output.
    nd = len(shape_tuple)

    # If the axis value is impossible for the given size array then
    # just fake it like it was for axis 0.  This will stop compile errors
    # when it looks like it could be called from array_sum_axis but really
    # can't because that routine checks the axis mismatch and raise an
    # exception.
    if axis_value >= nd:
        axis_value = 0

    # Calculate the type of the indexing tuple.  All the non-axis
    # dimensions have slice2 type and the axis dimension has int type.
    before = axis_value
    after  = nd - before - 1
    types_list = ([types.slice2_type] * before) +  \
                  [types.intp] +                   \
                 ([types.slice2_type] * after)

    # Creates the output type of the function.
    tupty = types.Tuple(types_list)
    # Defines the signature of the intrinsic.
    function_sig = tupty(shape_tuple, value, axis)

    def codegen(cgctx, builder, signature, args):
        lltupty = cgctx.get_value_type(tupty)
        # Create an empty indexing tuple.
        tup = cgutils.get_null_value(lltupty)

        # We only need value of the axis dimension here.
        # The rest are constants defined above.
        [_, value_arg, _] = args

        def create_full_slice():
            return slice(None, None)

        # loop to fill the tuple with slice(None,None) before
        # the axis dimension.

        # compile and call create_full_slice
        slice_data = cgctx.compile_internal(builder, create_full_slice,
                                            types.slice2_type(),
                                            [])
        for i in range(0, axis_value):
            tup = builder.insert_value(tup, slice_data, i)

        # Add the axis dimension 'value'.
        tup = builder.insert_value(tup, value_arg, axis_value)

        # loop to fill the tuple with slice(None,None) after
        # the axis dimension.
        for i in range(axis_value + 1, nd):
            tup = builder.insert_value(tup, slice_data, i)
        return tup

    return function_sig, codegen


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

@lower_builtin(np.sum, types.Array, types.intp)
@lower_builtin(np.sum, types.Array, types.Const)
@lower_builtin("array.sum", types.Array, types.intp)
@lower_builtin("array.sum", types.Array, types.Const)
def array_sum_axis(context, builder, sig, args):
    """
    The third parameter to gen_index_tuple that generates the indexing
    tuples has to be a const so we can't just pass "axis" through since
    that isn't const.  We can check for specific values and have
    different instances that do take consts.  Supporting axis summation
    only up to the fourth dimension for now.
    """
    # typing/arraydecl.py:sum_expand defines the return type for sum with axis.
    # It is one dimension less than the input array.
    zero = sig.return_type.dtype(0)

    [ty_array, ty_axis] = sig.args
    is_axis_const = False
    const_axis_val = 0
    if isinstance(ty_axis, types.Const):
        # this special-cases for constant axis
        const_axis_val = ty_axis.value
        # fix negative axis
        if const_axis_val < 0:
            const_axis_val = ty_array.ndim + const_axis_val
        if const_axis_val < 0 or const_axis_val > ty_array.ndim:
            raise ValueError("'axis' entry is out of bounds")

        ty_axis = context.typing_context.resolve_value_type(const_axis_val)
        axis_val = context.get_constant(ty_axis, const_axis_val)
        # rewrite arguments
        args = args[0], axis_val
        # rewrite sig
        sig = sig.replace(args=[ty_array, ty_axis])
        is_axis_const = True

    def array_sum_impl_axis(arr, axis):
        ndim = arr.ndim

        if not is_axis_const:
            # Catch where axis is negative or greater than 3.
            if axis < 0 or axis > 3:
                raise ValueError("Numba does not support sum with axis"
                                 "parameter outside the range 0 to 3.")

        # Catch the case where the user misspecifies the axis to be
        # more than the number of the array's dimensions.
        if axis >= ndim:
            raise ValueError("axis is out of bounds for array")

        # Convert the shape of the input array to a list.
        ashape = list(arr.shape)
        # Get the length of the axis dimension.
        axis_len = ashape[axis]
        # Remove the axis dimension from the list of dimensional lengths.
        ashape.pop(axis)
        # Convert this shape list back to a tuple using above intrinsic.
        ashape_without_axis = _create_tuple_result_shape(ashape, arr.shape)
        # Tuple needed here to create output array with correct size.
        result = np.full(ashape_without_axis, zero, type(zero))

        # Iterate through the axis dimension.
        for axis_index in range(axis_len):
            if is_axis_const:
                # constant specialized version works for any valid axis value
                index_tuple_generic = _gen_index_tuple(arr.shape, axis_index,
                                                       const_axis_val)
                result += arr[index_tuple_generic]
            else:
                # Generate a tuple used to index the input array.
                # The tuple is ":" in all dimensions except the axis
                # dimension where it is "axis_index".
                if axis == 0:
                    index_tuple1 = _gen_index_tuple(arr.shape, axis_index, 0)
                    result += arr[index_tuple1]
                elif axis == 1:
                    index_tuple2 = _gen_index_tuple(arr.shape, axis_index, 1)
                    result += arr[index_tuple2]
                elif axis == 2:
                    index_tuple3 = _gen_index_tuple(arr.shape, axis_index, 2)
                    result += arr[index_tuple3]
                elif axis == 3:
                    index_tuple4 = _gen_index_tuple(arr.shape, axis_index, 3)
                    result += arr[index_tuple4]

        return result

    res = context.compile_internal(builder, array_sum_impl_axis, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

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
        @register_jitable
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

if numpy_version >= (1, 10):
    @overload(np.nanprod)
    def np_nanprod(a):
        if not isinstance(a, types.Array):
            return
        if isinstance(a.dtype, types.Integer):
            retty = types.intp
        else:
            retty = a.dtype
        one = retty(1)
        isnan = get_isnan(a.dtype)

        def nanprod_impl(arr):
            c = one
            for view in np.nditer(arr):
                v = view.item()
                if not isnan(v):
                    c *= v
            return c

        return nanprod_impl

#----------------------------------------------------------------------------
# Median and partitioning

@register_jitable
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
    j = high - 1
    while True:
        while i < high and A[i] < pivot:
            i += 1
        while j >= low and pivot < A[j]:
            j -= 1
        if i >= j:
            break
        A[i], A[j] = A[j], A[i]
        i += 1
        j -= 1
    # Put the pivot back in its final place (all items before `i`
    # are smaller than the pivot, all items at/after `i` are larger)
    A[i], A[high] = A[high], A[i]
    return i

@register_jitable
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

@register_jitable
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

@register_jitable
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

@register_jitable
def _collect_percentiles_inner(a, q):
    n = len(a)

    if n == 1:
        # single element array; output same for all percentiles
        out = np.full(len(q), a[0], dtype=np.float64)
    else:
        out = np.empty(len(q), dtype=np.float64)
        for i in range(len(q)):
            percentile = q[i]

            # bypass pivoting where requested percentile is 100
            if percentile == 100:
                val = np.max(a)
                # heuristics to handle infinite values a la NumPy
                if ~np.all(np.isfinite(a)):
                    if ~np.isfinite(val):
                        val = np.nan

            # bypass pivoting where requested percentile is 0
            elif percentile == 0:
                val = np.min(a)
                # convoluted heuristics to handle infinite values a la NumPy
                if ~np.all(np.isfinite(a)):
                    num_pos_inf = np.sum(a == np.inf)
                    num_neg_inf = np.sum(a == -np.inf)
                    num_finite = n - (num_neg_inf + num_pos_inf)
                    if num_finite == 0:
                        val = np.nan
                    if num_pos_inf == 1 and n == 2:
                        val = np.nan
                    if num_neg_inf > 1:
                        val = np.nan
                    if num_finite == 1:
                        if num_pos_inf > 1:
                            if num_neg_inf != 1:
                                val = np.nan

            else:
                # linear interp between closest ranks
                rank = 1 + (n - 1) * np.true_divide(percentile, 100.0)
                f = math.floor(rank)
                m = rank - f
                lower, upper = _select_two(a, k=int(f - 1), low=0, high=(n - 1))
                val = lower * (1 - m) + upper * m
            out[i] = val

    return out

@register_jitable
def _can_collect_percentiles(a, nan_mask, skip_nan):
    if skip_nan:
        a = a[~nan_mask]
        if len(a) == 0:
            return False  # told to skip nan, but no elements remain
    else:
        if np.any(nan_mask):
            return False  # told *not* to skip nan, but nan encountered

    if len(a) == 1:  # single element array
        val = a[0]
        return np.isfinite(val)  # can collect percentiles if element is finite
    else:
        return True

@register_jitable
def _collect_percentiles(a, q, skip_nan=False):
    if np.any(np.isnan(q)) or np.any(q < 0) or np.any(q > 100):
        raise ValueError('Percentiles must be in the range [0,100]')

    temp_arry = a.flatten()
    nan_mask = np.isnan(temp_arry)

    if _can_collect_percentiles(temp_arry, nan_mask, skip_nan):
        temp_arry = temp_arry[~nan_mask]
        out = _collect_percentiles_inner(temp_arry, q)
    else:
        out = np.full(len(q), np.nan)

    return out

def _np_percentile_impl(a, q, skip_nan):
    def np_percentile_q_scalar_impl(a, q):
        percentiles = np.array([q])
        return _collect_percentiles(a, percentiles, skip_nan=skip_nan)[0]

    def np_percentile_q_sequence_impl(a, q):
        percentiles = np.array(q)
        return _collect_percentiles(a, percentiles, skip_nan=skip_nan)

    def np_percentile_q_array_impl(a, q):
        return _collect_percentiles(a, q, skip_nan=skip_nan)

    if isinstance(q, (types.Float, types.Integer, types.Boolean)):
        return np_percentile_q_scalar_impl

    elif isinstance(q, (types.Tuple, types.Sequence)):
        return np_percentile_q_sequence_impl

    elif isinstance(q, types.Array):
        return np_percentile_q_array_impl

if numpy_version >= (1, 10):
    @overload(np.percentile)
    def np_percentile(a, q):
        # Note: np.percentile behaviour in the case of an array containing one or
        # more NaNs was changed in numpy 1.10 to return an array of np.NaN of
        # length equal to q, hence version guard.
        return _np_percentile_impl(a, q, skip_nan=False)

if numpy_version >= (1, 11):
    @overload(np.nanpercentile)
    def np_nanpercentile(a, q):
        # Note: np.nanpercentile return type in the case of an all-NaN slice
        # was changed in 1.11 to be an array of np.NaN of length equal to q,
        # hence version guard.
        return _np_percentile_impl(a, q, skip_nan=True)

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

            # all NaNs
            if n == 0:
                return np.nan

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


@overload(np.real)
def np_real(a):
    def np_real_impl(a):
        return a.real

    return np_real_impl


@overload(np.imag)
def np_imag(a):
    def np_imag_impl(a):
        return a.imag

    return np_imag_impl


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

        @register_jitable
        def validate_inputs(a, weights):
            if len(a) != len(weights):
                raise ValueError("bincount(): weights and list don't have the same length")

        @register_jitable
        def count_item(out, idx, val, weights):
            out[val] += weights[idx]

    else:
        out_dtype = types.intp

        @register_jitable
        def validate_inputs(a, weights):
            pass

        @register_jitable
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

def _searchsorted(func):
    def searchsorted_inner(a, v):
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
            if func(a[mid], (v)):
                # mid is too low => go up
                lo = mid + 1
            else:
                # mid is too high, or is a NaN => go down
                hi = mid
        return lo
    return searchsorted_inner

_lt = register_jitable(lambda x, y: x < y)
_le = register_jitable(lambda x, y: x <= y)
_searchsorted_left = register_jitable(_searchsorted(_lt))
_searchsorted_right = register_jitable(_searchsorted(_le))

@overload(np.searchsorted)
def searchsorted(a, v, side='left'):
    side_val = getattr(side, 'value', side)
    if side_val == 'left':
        loop_impl = _searchsorted_left
    elif side_val == 'right':
        loop_impl = _searchsorted_right
    else:
        raise ValueError("Invalid value given for 'side': %s" % side_val)

    if isinstance(v, types.Array):
        # N-d array and output
        def searchsorted_impl(a, v, side='left'):
            out = np.empty(v.shape, np.intp)
            for view, outview in np.nditer((v, out)):
                index = loop_impl(a, view.item())
                outview.itemset(index)
            return out

    elif isinstance(v, types.Sequence):
        # 1-d sequence and output
        def searchsorted_impl(a, v, side='left'):
            out = np.empty(len(v), np.intp)
            for i in range(len(v)):
                out[i] = loop_impl(a, v[i])
            return out
    else:
        # Scalar value and output
        # Note: NaNs come last in Numpy-sorted arrays
        def searchsorted_impl(a, v, side='left'):
            return loop_impl(a, v)

    return searchsorted_impl

@overload(np.digitize)
def np_digitize(x, bins, right=False):
    @register_jitable
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

    @register_jitable
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

    @register_jitable
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


# Create np.finfo, np.iinfo and np.MachAr
# machar
_mach_ar_supported = ('ibeta', 'it', 'machep', 'eps', 'negep', 'epsneg',
                      'iexp', 'minexp', 'xmin', 'maxexp', 'xmax', 'irnd',
                      'ngrd', 'epsilon', 'tiny', 'huge', 'precision',
                      'resolution',)
MachAr = namedtuple('MachAr', _mach_ar_supported)

# Do not support MachAr field
# finfo
_finfo_supported = ('eps', 'epsneg', 'iexp', 'machep', 'max', 'maxexp', 'min',
                    'minexp', 'negep', 'nexp', 'nmant', 'precision',
                    'resolution', 'tiny',)
if numpy_version >= (1, 12):
    _finfo_supported = ('bits',) + _finfo_supported

finfo = namedtuple('finfo', _finfo_supported)

# iinfo
_iinfo_supported = ('min', 'max')
if numpy_version >= (1, 12):
    _iinfo_supported = _iinfo_supported + ('bits',)

iinfo = namedtuple('iinfo', _iinfo_supported)

@overload(np.MachAr)
def MachAr_impl():
    f = np.MachAr()
    _mach_ar_data = tuple([getattr(f, x) for x in _mach_ar_supported])
    def impl():
        return MachAr(*_mach_ar_data)
    return impl

def generate_xinfo(np_func, container, attr):
    @overload(np_func)
    def xinfo_impl(arg):
        nbty = getattr(arg, 'dtype', arg)
        f = np_func(as_dtype(nbty))
        data = tuple([getattr(f, x) for x in attr])
        def impl(arg):
            return container(*data)
        return impl

generate_xinfo(np.finfo, finfo, _finfo_supported)
generate_xinfo(np.iinfo, iinfo, _iinfo_supported)

def _get_inner_prod(dta, dtb):
    # gets an inner product implementation, if both types are float then
    # BLAS is used else a local function

    @register_jitable
    def _innerprod(a, b):
        acc = 0
        for i in range(len(a)):
            acc = acc + a[i] * b[i]
        return acc

    # no BLAS... use local function regardless
    if not _HAVE_BLAS:
        return _innerprod

    flty = types.real_domain | types.complex_domain
    floats = dta in flty and dtb in flty
    if not floats:
        return _innerprod
    else:
        a_dt = as_dtype(dta)
        b_dt = as_dtype(dtb)
        dt = np.promote_types(a_dt, b_dt)

        @register_jitable
        def _dot_wrap(a, b):
            return np.dot(a.astype(dt), b.astype(dt))
        return _dot_wrap

def _assert_1d(a, func_name):
    if isinstance(a, types.Array):
        if not a.ndim <= 1:
            raise TypingError("%s() only supported on 1D arrays " % func_name)

def _np_correlate_core(ap1, ap2, mode, direction):
    pass


class _corr_conv_Mode(IntEnum):
    """
    Enumerated modes for correlate/convolve as per:
    https://github.com/numpy/numpy/blob/ac6b1a902b99e340cf7eeeeb7392c91e38db9dd8/numpy/core/numeric.py#L862-L870
    """
    VALID = 0
    SAME = 1
    FULL = 2


@overload(_np_correlate_core)
def _np_correlate_core_impl(ap1, ap2, mode, direction):
    a_dt = as_dtype(ap1.dtype)
    b_dt = as_dtype(ap2.dtype)
    dt = np.promote_types(a_dt, b_dt)
    innerprod = _get_inner_prod(ap1.dtype, ap2.dtype)

    Mode = _corr_conv_Mode

    def impl(ap1, ap2, mode, direction):
        # Implementation loosely based on `_pyarray_correlate` from
        # https://github.com/numpy/numpy/blob/3bce2be74f228684ca2895ad02b63953f37e2a9d/numpy/core/src/multiarray/multiarraymodule.c#L1191
        # For "Mode":
        # Convolve uses 'full' by default, this is denoted by the number 2
        # Correlate uses 'valid' by default, this is denoted by the number 0
        # For "direction", +1 to write the return values out in order 0->N
        # -1 to write them out N->0.

        if not (mode == Mode.VALID or mode == Mode.FULL):
            raise ValueError("Invalid mode")

        n1 = len(ap1)
        n2 = len(ap2)
        length = n1
        n = n2
        if mode == Mode.VALID: # mode == valid == 0, correlate default
            length = length - n + 1
            n_left = 0
            n_right = 0
        elif mode == Mode.FULL: # mode == full == 2, convolve default
            n_right = n - 1
            n_left = n - 1
            length = length + n - 1
        else:
            raise ValueError("Invalid mode")

        ret = np.zeros(length, dt)
        n = n - n_left

        if direction == 1:
            idx = 0
            inc = 1
        elif direction == -1:
            idx = length - 1
            inc = -1
        else:
            raise ValueError("Invalid direction")

        for i in range(n_left):
            ret[idx] = innerprod(ap1[:idx + 1], ap2[-(idx + 1):])
            idx = idx + inc

        for i in range(n1 - n2 + 1):
            ret[idx] = innerprod(ap1[i : i + n2], ap2)
            idx = idx + inc

        for i in range(n_right, 0, -1):
            ret[idx] = innerprod(ap1[-i:], ap2[:i])
            idx = idx + inc
        return ret

    return impl

@overload(np.correlate)
def _np_correlate(a, v):
    _assert_1d(a, 'np.correlate')
    _assert_1d(v, 'np.correlate')

    @register_jitable
    def op_conj(x):
        return np.conj(x)

    @register_jitable
    def op_nop(x):
        return x

    Mode = _corr_conv_Mode

    if a.dtype in types.complex_domain:
        if v.dtype in types.complex_domain:
            a_op = op_nop
            b_op = op_conj
        else:
            a_op = op_nop
            b_op = op_nop
    else:
        if v.dtype in types.complex_domain:
            a_op = op_nop
            b_op = op_conj
        else:
            a_op = op_conj
            b_op = op_nop

    def impl(a, v):
        if len(a) < len(v):
            return _np_correlate_core(b_op(v), a_op(a), Mode.VALID, -1)
        else:
            return _np_correlate_core(a_op(a), b_op(v), Mode.VALID, 1)

    return impl

@overload(np.convolve)
def np_convolve(a, v):
    _assert_1d(a, 'np.convolve')
    _assert_1d(v, 'np.convolve')

    Mode = _corr_conv_Mode

    def impl(a, v):
        la = len(a)
        lv = len(v)

        if la == 0:
            raise ValueError("'a' cannot be empty")
        if lv == 0:
            raise ValueError("'v' cannot be empty")

        if la < lv:
            return _np_correlate_core(v, a[::-1], Mode.FULL, 1)
        else:
            return _np_correlate_core(a, v[::-1], Mode.FULL, 1)

    return impl
