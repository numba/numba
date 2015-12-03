"""
Implementation of math operations on Array objects.
"""

from __future__ import print_function, absolute_import, division

import math

from llvmlite import ir
import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Constant, Type

import numpy
from numba import types, cgutils, typing
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (builtin, implement, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.typing import signature
from .arrayobj import make_array, load_item, store_item, _empty_nd_impl


#----------------------------------------------------------------------------
# Stats and aggregates

@builtin
@implement(numpy.sum, types.Kind(types.Array))
@implement("array.sum", types.Kind(types.Array))
def array_sum(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_sum_impl(arr):
        c = zero
        for v in arr.flat:
            c += v
        return c

    res = context.compile_internal(builder, array_sum_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement(numpy.prod, types.Kind(types.Array))
@implement("array.prod", types.Kind(types.Array))
def array_prod(context, builder, sig, args):

    def array_prod_impl(arr):
        c = 1
        for v in arr.flat:
            c *= v
        return c

    res = context.compile_internal(builder, array_prod_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement(numpy.cumsum, types.Kind(types.Array))
@implement("array.cumsum", types.Kind(types.Array))
def array_cumsum(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)
    zero = scalar_dtype(0)

    def array_cumsum_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = numpy.empty(size, dtype)
        c = zero
        for idx, v in enumerate(arr.flat):
            c += v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumsum_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)



@builtin
@implement(numpy.cumprod, types.Kind(types.Array))
@implement("array.cumprod", types.Kind(types.Array))
def array_cumprod(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)

    def array_cumprod_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = numpy.empty(size, dtype)
        c = 1
        for idx, v in enumerate(arr.flat):
            c *= v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumprod_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.mean, types.Kind(types.Array))
@implement("array.mean", types.Kind(types.Array))
def array_mean(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_mean_impl(arr):
        # Can't use the naive `arr.sum() / arr.size`, as it would return
        # a wrong result on integer sum overflow.
        c = zero
        for v in arr.flat:
            c += v
        return c / arr.size

    res = context.compile_internal(builder, array_mean_impl, sig, args,
                                   locals=dict(c=sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.var, types.Kind(types.Array))
@implement("array.var", types.Kind(types.Array))
def array_var(context, builder, sig, args):
    def array_var_impl(arry):
        # Compute the mean
        m = arry.mean()

        # Compute the sum of square diffs
        ssd = 0
        for v in arry.flat:
            ssd += (v - m) ** 2
        return ssd / arry.size

    res = context.compile_internal(builder, array_var_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.std, types.Kind(types.Array))
@implement("array.std", types.Kind(types.Array))
def array_std(context, builder, sig, args):
    def array_std_impl(arry):
        return arry.var() ** 0.5
    res = context.compile_internal(builder, array_std_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.min, types.Kind(types.Array))
@implement("array.min", types.Kind(types.Array))
def array_min(context, builder, sig, args):
    ty = sig.args[0].dtype
    if isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
        # NaT is smaller than every other value, but it is
        # ignored as far as min() is concerned.
        nat = ty('NaT')

        def array_min_impl(arry):
            min_value = nat
            it = arry.flat
            for v in it:
                if v != nat:
                    min_value = v
                    break

            for v in it:
                if v != nat and v < min_value:
                    min_value = v
            return min_value

    else:
        def array_min_impl(arry):
            for v in arry.flat:
                min_value = v
                break

            for v in arry.flat:
                if v < min_value:
                    min_value = v
            return min_value
    res = context.compile_internal(builder, array_min_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement(numpy.max, types.Kind(types.Array))
@implement("array.max", types.Kind(types.Array))
def array_max(context, builder, sig, args):
    def array_max_impl(arry):
        for v in arry.flat:
            max_value = v
            break

        for v in arry.flat:
            if v > max_value:
                max_value = v
        return max_value
    res = context.compile_internal(builder, array_max_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement(numpy.argmin, types.Kind(types.Array))
@implement("array.argmin", types.Kind(types.Array))
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


@builtin
@implement(numpy.argmax, types.Kind(types.Array))
@implement("array.argmax", types.Kind(types.Array))
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


@builtin
@implement(numpy.median, types.Kind(types.Array))
def array_median(context, builder, sig, args):

    def partition(A, low, high):
        mid = (low+high) // 2
        # median of three {low, middle, high}
        LM = A[low] <= A[mid]
        MH = A[mid] <= A[high]
        LH = A[low] <= A[high]

        if LM == MH:
            median3 = mid
        elif LH != LM:
            median3 = low
        else:
            median3 = high

        # choose median3 as the pivot
        A[high], A[median3] = A[median3], A[high]

        x = A[high]
        i = low
        for j in range(low, high):
            if A[j] <= x:
                A[i], A[j] = A[j], A[i]
                i += 1
        A[i], A[high] = A[high], A[i]
        return i

    sig_partition = typing.signature(types.intp, *(sig.args[0], types.intp, types.intp))
    _partition = context.compile_subroutine(builder, partition, sig_partition)

    def select(arry, k):
        n = arry.shape[0]
        # XXX: assuming flat array till array.flatten is implemented
        # temp_arry = arry.flatten()
        temp_arry = arry.copy()
        high = n-1
        low = 0
        # NOTE: high is inclusive
        i = _partition(temp_arry, low, high)
        while i != k:
            if i < k:
                low = i+1
                i = _partition(temp_arry, low, high)
            else:
                high = i-1
                i = _partition(temp_arry, low, high)
        return temp_arry[k]

    sig_select = typing.signature(sig.args[0].dtype, *(sig.args[0], types.intp))
    _select = context.compile_subroutine(builder, select, sig_select)

    def median(arry):
        n = arry.shape[0]
        if n % 2 == 0:
            return (_select(arry, n//2 - 1) + _select(arry, n//2))/2
        else:
            return _select(arry, n//2)

    res = context.compile_internal(builder, median, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


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

@builtin
@implement(numpy.round, types.Kind(types.Float))
def scalar_round_unary(context, builder, sig, args):
    res =  _np_round_float(context, builder, sig.args[0], args[0])
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Integer))
def scalar_round_unary(context, builder, sig, args):
    res = args[0]
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Complex))
def scalar_round_unary_complex(context, builder, sig, args):
    fltty = sig.args[0].underlying_float
    cplx_cls = context.make_complex(sig.args[0])
    z = cplx_cls(context, builder, args[0])
    z.real = _np_round_float(context, builder, fltty, z.real)
    z.imag = _np_round_float(context, builder, fltty, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Float), types.Kind(types.Integer))
@implement(numpy.round, types.Kind(types.Integer), types.Kind(types.Integer))
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
            return (numpy.round(y) / pow2) / pow1

        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return numpy.round(y) * pow1

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Complex), types.Kind(types.Integer))
def scalar_round_binary_complex(context, builder, sig, args):
    def round_ndigits(z, ndigits):
        return complex(numpy.round(z.real, ndigits),
                       numpy.round(z.imag, ndigits))

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.round, types.Kind(types.Array), types.Kind(types.Integer),
           types.Kind(types.Array))
def array_round(context, builder, sig, args):
    def array_round_impl(arr, decimals, out):
        if arr.shape != out.shape:
            raise ValueError("invalid output shape")
        for index, val in numpy.ndenumerate(arr):
            out[index] = numpy.round(val, decimals)
        return out

    res = context.compile_internal(builder, array_round_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.sinc, types.Kind(types.Array))
def array_sinc(context, builder, sig, args):
    def array_sinc_impl(arr):
        out = numpy.zeros_like(arr)
        for index, val in numpy.ndenumerate(arr):
            out[index] = numpy.sinc(val)
        return out
    res = context.compile_internal(builder, array_sinc_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.sinc, types.Kind(types.Number))
def scalar_sinc(context, builder, sig, args):
    scalar_dtype = sig.return_type
    def scalar_sinc_impl(val):
        if val == 0.e0: # to match np impl
            val = 1e-20
        val *= numpy.pi # np sinc is the normalised variant
        return numpy.sin(val)/val
    res = context.compile_internal(builder, scalar_sinc_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.angle, types.Kind(types.Number))
def scalar_angle(context, builder, sig, args):
    def scalar_angle_impl(val):
          return numpy.arctan2(val.imag, val.real)
    res = context.compile_internal(builder, scalar_angle_impl,
                                      sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.angle, types.Kind(types.Number), types.Kind(types.Boolean))
def scalar_angle_kwarg(context, builder, sig, args):
    def scalar_angle_impl(val, deg=False):
        if deg:
            scal = 180/numpy.pi
            return numpy.arctan2(val.imag, val.real) * scal
        else:
            return numpy.arctan2(val.imag, val.real)
    res = context.compile_internal(builder, scalar_angle_impl,
                                      sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.angle, types.Kind(types.Array))
@implement(numpy.angle, types.Kind(types.Array), types.Kind(types.Boolean))
def array_angle_kwarg(context, builder, sig, args):
    arg = sig.args[0]
    if isinstance(arg.dtype, types.Complex):
        retty = arg.dtype.underlying_float
    else:
        retty = arg.dtype
    def array_angle_impl(arr, deg=False):
        out = numpy.zeros_like(arr, dtype=retty)
        for index, val in numpy.ndenumerate(arr):
            out[index] = numpy.angle(val, deg)
        return out
    res = context.compile_internal(builder, array_angle_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.nonzero, types.Kind(types.Array))
@implement("array.nonzero", types.Kind(types.Array))
@implement(numpy.where, types.Kind(types.Array))
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
            res = numpy.empty_like(x)
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
            res = numpy.empty_like(x)
            for idx, c in numpy.ndenumerate(cond):
                res[idx] = x[idx] if c else y[idx]
            return res

    res = context.compile_internal(builder, where_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.where, types.Any, types.Any, types.Any)
def any_where(context, builder, sig, args):
    cond = sig.args[0]
    if isinstance(cond, types.Array):
        return array_where(context, builder, sig, args)

    def scalar_where_impl(cond, x, y):
        """
        np.where(scalar, scalar, scalar): return a 0-dim array
        """
        scal = x if cond else y
        # This is the equivalent of numpy.full_like(scal, scal),
        # for compatibility with Numpy < 1.8
        arr = numpy.empty_like(scal)
        arr[()] = scal
        return arr

    res = context.compile_internal(builder, scalar_where_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


#----------------------------------------------------------------------------
# Linear algebra

ll_char = ir.IntType(8)
ll_char_p = ll_char.as_pointer()
ll_void_p = ll_char_p
intp_t = cgutils.intp_t

def get_blas_kind(dtype):
    return {
        types.float32: 's',
        types.float64: 'd',
        types.complex64: 'c',
        types.complex128: 'z',
        }[dtype]

def ensure_blas():
    try:
        import scipy.linalg.cython_blas
    except ImportError:
        raise ImportError("scipy 0.16+ is required for linear algebra")

def make_constant_slot(context, builder, ty, val):
    const = context.get_constant_generic(builder, ty, val)
    return cgutils.alloca_once_value(builder, const)

def check_c_int(context, builder, n):
    _maxint = 2**31 - 1

    def impl(n):
        if n > _maxint:
            raise OverflowError("array size too large to fit in C int")

    context.compile_internal(builder, impl,
                             signature(types.none, types.intp), (n,))

def check_blas_return(context, builder, res):
    with builder.if_then(cgutils.is_not_null(builder, res), likely=False):
        # Those errors shouldn't happen, it's easier to just abort the process
        pyapi = context.get_python_api(builder)
        pyapi.gil_ensure()
        pyapi.fatal_error("BLAS wrapper returned with an error")


def dot_2_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
    def dot_impl(a, b):
        m, k = a.shape
        _k, n = b.shape
        out = numpy.empty((m, n), a.dtype)
        return numpy.dot(a, b, out)

    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """
    def dot_impl(a, b):
        m, = a.shape
        _m, n = b.shape
        out = numpy.empty((n, ), a.dtype)
        return numpy.dot(a, b, out)

    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

def dot_2_mv(context, builder, sig, args):
    """
    np.dot(matrix, vector)
    """
    def dot_impl(a, b):
        m, n = a.shape
        _n, = b.shape
        out = numpy.empty((m, ), a.dtype)
        return numpy.dot(a, b, out)

    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

def dot_2_vv(context, builder, sig, args, conjugate=False):
    """
    np.dot(vector, vector)
    np.vdot(vector, vector)
    """
    aty, bty = sig.args
    dtype = sig.return_type
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    n, = cgutils.unpack_tuple(builder, a.shape)

    def check_args(a, b):
        m, = a.shape
        n, = b.shape
        if m != n:
            raise ValueError("incompatible array sizes for np.dot(a, b) "
                             "(vector * vector)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    check_c_int(context, builder, n)

    out = cgutils.alloca_once(builder, context.get_value_type(dtype))

    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char, ll_char, intp_t,        # kind, conjugate, n
                            ll_void_p, ll_void_p, ll_void_p, # a, b, out
                           ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxdot")

    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    conjugate = ir.Constant(ll_char, int(conjugate))

    res = builder.call(fn, (kind_val, conjugate, n,
                            builder.bitcast(a.data, ll_void_p),
                            builder.bitcast(b.data, ll_void_p),
                            builder.bitcast(out, ll_void_p)))
    check_blas_return(context, builder, res)

    return builder.load(out)


@builtin
@implement(numpy.dot, types.Kind(types.Array), types.Kind(types.Array))
@implement('@', types.Kind(types.Array), types.Kind(types.Array))
def dot_2(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """
    ensure_blas()

    ndims = [x.ndim for x in sig.args[:2]]
    if ndims == [2, 2]:
        return dot_2_mm(context, builder, sig, args)
    elif ndims == [2, 1]:
        return dot_2_mv(context, builder, sig, args)
    elif ndims == [1, 2]:
        return dot_2_vm(context, builder, sig, args)
    elif ndims == [1, 1]:
        return dot_2_vv(context, builder, sig, args)
    else:
        assert 0

@builtin
@implement(numpy.vdot, types.Kind(types.Array), types.Kind(types.Array))
def vdot(context, builder, sig, args):
    """
    np.vdot(a, b)
    """
    ensure_blas()

    return dot_2_vv(context, builder, sig, args, conjugate=True)


def dot_3_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix, out)
    np.dot(matrix, vector, out)
    """
    xty, yty, outty = sig.args
    assert outty == sig.return_type
    dtype = xty.dtype

    x = make_array(xty)(context, builder, args[0])
    y = make_array(yty)(context, builder, args[1])
    out = make_array(outty)(context, builder, args[2])
    x_shapes = cgutils.unpack_tuple(builder, x.shape)
    y_shapes = cgutils.unpack_tuple(builder, y.shape)
    out_shapes = cgutils.unpack_tuple(builder, out.shape)
    if xty.ndim < yty.ndim:
        # Vector * matrix
        # Asked for x * y, we will compute y.T * x
        mty = yty
        m, n = m_shapes = y_shapes
        do_trans = yty.layout == 'F'
        m_data, v_data = y.data, x.data

        def check_args(a, b, out):
            m, = a.shape
            _m, n = b.shape
            if m != _m:
                raise ValueError("incompatible array sizes for np.dot(a, b) "
                                 "(vector * matrix)")
            if out.shape != (n,):
                raise ValueError("incompatible output array size for np.dot(a, b, out) "
                                 "(vector * matrix)")
    else:
        # Matrix * vector
        # We will compute x * y
        mty = xty
        m, n = m_shapes = x_shapes
        do_trans = xty.layout == 'C'
        m_data, v_data = x.data, y.data

        def check_args(a, b, out):
            m, _n= a.shape
            n, = b.shape
            if n != _n:
                raise ValueError("incompatible array sizes for np.dot(a, b) "
                                 "(matrix * vector)")
            if out.shape != (m,):
                raise ValueError("incompatible output array size for np.dot(a, b, out) "
                                 "(matrix * vector)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    check_c_int(context, builder, m)
    check_c_int(context, builder, n)

    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char, ll_char_p,               # kind, trans
                            intp_t, intp_t,                   # m, n
                            ll_void_p, ll_void_p, intp_t,     # alpha, a, lda
                            ll_void_p, ll_void_p, ll_void_p,  # x, beta, y
                           ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgemv")

    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)

    if mty.layout == 'F':
        lda = m_shapes[0]
    else:
        m, n = n, m
        lda = m_shapes[1]

    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    trans = context.insert_const_string(builder.module,
                                        "t" if do_trans else "n")

    res = builder.call(fn, (kind_val, trans, m, n,
                            builder.bitcast(alpha, ll_void_p),
                            builder.bitcast(m_data, ll_void_p), lda,
                            builder.bitcast(v_data, ll_void_p),
                            builder.bitcast(beta, ll_void_p),
                            builder.bitcast(out.data, ll_void_p)))
    check_blas_return(context, builder, res)

    return impl_ret_borrowed(context, builder, sig.return_type, out._getvalue())


def dot_3_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix, out)
    """
    xty, yty, outty = sig.args
    assert outty == sig.return_type
    dtype = xty.dtype

    x = make_array(xty)(context, builder, args[0])
    y = make_array(yty)(context, builder, args[1])
    out = make_array(outty)(context, builder, args[2])
    x_shapes = cgutils.unpack_tuple(builder, x.shape)
    y_shapes = cgutils.unpack_tuple(builder, y.shape)
    out_shapes = cgutils.unpack_tuple(builder, out.shape)
    m, k = x_shapes
    _k, n = y_shapes

    def check_args(a, b, out):
        m, k = a.shape
        _k, n = b.shape
        if k != _k:
            raise ValueError("incompatible array sizes for np.dot(a, b) "
                             "(matrix * matrix)")
        if out.shape != (m, n):
            raise ValueError("incompatible output array size for np.dot(a, b, out) "
                             "(matrix * matrix)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    check_c_int(context, builder, m)
    check_c_int(context, builder, k)
    check_c_int(context, builder, n)

    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char,                       # kind
                            ll_char_p, ll_char_p,          # transa, transb
                            intp_t, intp_t, intp_t,        # m, n, k
                            ll_void_p, ll_void_p, intp_t,  # alpha, a, lda
                            ll_void_p, intp_t, ll_void_p,  # b, ldb, beta
                            ll_void_p, intp_t,             # c, ldc
                           ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgemm")

    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)

    trans = context.insert_const_string(builder.module, "t")
    notrans = context.insert_const_string(builder.module, "n")

    # Since out is C-contiguous, compute a * b = y.T * x.T
    assert outty.layout == 'C'

    def get_array_param(ty, shapes, data):
        return (
                # Transpose if layout different from result's
                notrans if ty.layout == outty.layout else trans,
                # Size of the inner dimension in physical array order
                shapes[1] if ty.layout == 'C' else shapes[0],
                # The data pointer, unit-less
                builder.bitcast(data, ll_void_p),
                )

    transa, lda, data_a = get_array_param(yty, y_shapes, y.data)
    transb, ldb, data_b = get_array_param(xty, x_shapes, x.data)
    _, ldc, data_c = get_array_param(outty, out_shapes, out.data)

    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))

    res = builder.call(fn, (kind_val, transa, transb, n, m, k,
                            builder.bitcast(alpha, ll_void_p), data_a, lda,
                            data_b, ldb, builder.bitcast(beta, ll_void_p),
                            data_c, ldc))
    check_blas_return(context, builder, res)

    return impl_ret_borrowed(context, builder, sig.return_type, out._getvalue())


@builtin
@implement(numpy.dot, types.Kind(types.Array), types.Kind(types.Array),
           types.Kind(types.Array))
def dot_3(context, builder, sig, args):
    """
    np.dot(a, b, out)
    """
    ensure_blas()

    ndims = set(x.ndim for x in sig.args[:2])
    if ndims == set([2]):
        return dot_3_mm(context, builder, sig, args)
    elif ndims == set([1, 2]):
        return dot_3_vm(context, builder, sig, args)
    else:
        assert 0
