"""
Implementation of linear algebra operations.
"""

from __future__ import print_function, absolute_import, division

import contextlib

from llvmlite import ir
import numpy


from numba import jit, types, cgutils

from numba.targets.imputils import (lower_builtin, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.typing import signature
from numba.extending import overload
from numba.numpy_support import version as numpy_version
from numba import types
from .arrayobj import make_array, _empty_nd_impl, array_copy
from ..errors import TypingError

ll_char = ir.IntType(8)
ll_char_p = ll_char.as_pointer()
ll_void_p = ll_char_p
ll_intc = ir.IntType(32)
ll_intc_p = ll_intc.as_pointer()
intp_t = cgutils.intp_t


_blas_kinds = {
    types.float32: 's',
    types.float64: 'd',
    types.complex64: 'c',
    types.complex128: 'z',
}


def get_blas_kind(dtype, func_name="<BLAS function>"):
    kind = _blas_kinds.get(dtype)
    if kind is None:
        raise TypeError("unsupported dtype for %s()" % (func_name,))
    return kind


def ensure_blas():
    try:
        import scipy.linalg.cython_blas
    except ImportError:
        raise ImportError("scipy 0.16+ is required for linear algebra")


def ensure_lapack():
    try:
        import scipy.linalg.cython_lapack
    except ImportError:
        raise ImportError("scipy 0.16+ is required for linear algebra")


def make_constant_slot(context, builder, ty, val):
    const = context.get_constant_generic(builder, ty, val)
    return cgutils.alloca_once_value(builder, const)


@contextlib.contextmanager
def make_contiguous(context, builder, sig, args):
    """
    Ensure that all array arguments are contiguous, if necessary by
    copying them.
    A new (sig, args) tuple is yielded.
    """
    newtys = []
    newargs = []
    copies = []
    for ty, val in zip(sig.args, args):
        if not isinstance(ty, types.Array) or ty.layout in 'CF':
            newty, newval = ty, val
        else:
            newty = ty.copy(layout='C')
            copysig = signature(newty, ty)
            newval = array_copy(context, builder, copysig, (val,))
            copies.append((newty, newval))
        newtys.append(newty)
        newargs.append(newval)
    yield signature(sig.return_type, *newtys), tuple(newargs)
    for ty, val in copies:
        context.nrt_decref(builder, ty, val)


def check_c_int(context, builder, n):
    """
    Check whether *n* fits in a C `int`.
    """
    _maxint = 2**31 - 1

    def impl(n):
        if n > _maxint:
            raise OverflowError("array size too large to fit in C int")

    context.compile_internal(builder, impl,
                             signature(types.none, types.intp), (n,))


def check_blas_return(context, builder, res):
    """
    Check the integer error return from one of the BLAS wrappers in
    _helperlib.c.
    """
    with builder.if_then(cgutils.is_not_null(builder, res), likely=False):
        # Those errors shouldn't happen, it's easier to just abort the process
        pyapi = context.get_python_api(builder)
        pyapi.gil_ensure()
        pyapi.fatal_error("BLAS wrapper returned with an error")


def check_lapack_return(context, builder, res):
    """
    Check the integer error return from one of the LAPACK wrappers in
    _helperlib.c.
    """
    with builder.if_then(cgutils.is_not_null(builder, res), likely=False):
        # Those errors shouldn't happen, it's easier to just abort the process
        pyapi = context.get_python_api(builder)
        pyapi.gil_ensure()
        pyapi.fatal_error("LAPACK wrapper returned with an error")


def call_xxdot(context, builder, conjugate, dtype,
               n, a_data, b_data, out_data):
    """
    Call the BLAS vector * vector product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char, ll_char, intp_t,    # kind, conjugate, n
                            ll_void_p, ll_void_p, ll_void_p,  # a, b, out
                            ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxdot")

    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    conjugate = ir.Constant(ll_char, int(conjugate))

    res = builder.call(fn, (kind_val, conjugate, n,
                            builder.bitcast(a_data, ll_void_p),
                            builder.bitcast(b_data, ll_void_p),
                            builder.bitcast(out_data, ll_void_p)))
    check_blas_return(context, builder, res)


def call_xxgemv(context, builder, do_trans,
                m_type, m_shapes, m_data, v_data, out_data):
    """
    Call the BLAS matrix * vector product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char, ll_char_p,               # kind, trans
                            intp_t, intp_t,                   # m, n
                            ll_void_p, ll_void_p, intp_t,     # alpha, a, lda
                            ll_void_p, ll_void_p, ll_void_p,  # x, beta, y
                            ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgemv")

    dtype = m_type.dtype
    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)

    if m_type.layout == 'F':
        m, n = m_shapes
        lda = m_shapes[0]
    else:
        n, m = m_shapes
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
                            builder.bitcast(out_data, ll_void_p)))
    check_blas_return(context, builder, res)


def call_xxgemm(context, builder,
                x_type, x_shapes, x_data,
                y_type, y_shapes, y_data,
                out_type, out_shapes, out_data):
    """
    Call the BLAS matrix * matrix product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_char,                       # kind
                            ll_char_p, ll_char_p,          # transa, transb
                            intp_t, intp_t, intp_t,        # m, n, k
                            ll_void_p, ll_void_p, intp_t,  # alpha, a, lda
                            ll_void_p, intp_t, ll_void_p,  # b, ldb, beta
                            ll_void_p, intp_t,             # c, ldc
                            ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgemm")

    m, k = x_shapes
    _k, n = y_shapes
    dtype = x_type.dtype
    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)

    trans = context.insert_const_string(builder.module, "t")
    notrans = context.insert_const_string(builder.module, "n")

    def get_array_param(ty, shapes, data):
        return (
            # Transpose if layout different from result's
            notrans if ty.layout == out_type.layout else trans,
            # Size of the inner dimension in physical array order
            shapes[1] if ty.layout == 'C' else shapes[0],
            # The data pointer, unit-less
            builder.bitcast(data, ll_void_p),
        )

    transa, lda, data_a = get_array_param(y_type, y_shapes, y_data)
    transb, ldb, data_b = get_array_param(x_type, x_shapes, x_data)
    _, ldc, data_c = get_array_param(out_type, out_shapes, out_data)

    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))

    res = builder.call(fn, (kind_val, transa, transb, n, m, k,
                            builder.bitcast(alpha, ll_void_p), data_a, lda,
                            data_b, ldb, builder.bitcast(beta, ll_void_p),
                            data_c, ldc))
    check_blas_return(context, builder, res)


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
    call_xxdot(context, builder, conjugate, dtype, n, a.data, b.data, out)
    return builder.load(out)


@lower_builtin(numpy.dot, types.Array, types.Array)
@lower_builtin('@', types.Array, types.Array)
def dot_2(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """
    ensure_blas()

    with make_contiguous(context, builder, sig, args) as (sig, args):
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


@lower_builtin(numpy.vdot, types.Array, types.Array)
def vdot(context, builder, sig, args):
    """
    np.vdot(a, b)
    """
    ensure_blas()

    with make_contiguous(context, builder, sig, args) as (sig, args):
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
        m_shapes = y_shapes
        do_trans = yty.layout == 'F'
        m_data, v_data = y.data, x.data

        def check_args(a, b, out):
            m, = a.shape
            _m, n = b.shape
            if m != _m:
                raise ValueError("incompatible array sizes for "
                                 "np.dot(a, b) (vector * matrix)")
            if out.shape != (n,):
                raise ValueError("incompatible output array size for "
                                 "np.dot(a, b, out) (vector * matrix)")
    else:
        # Matrix * vector
        # We will compute x * y
        mty = xty
        m_shapes = x_shapes
        do_trans = xty.layout == 'C'
        m_data, v_data = x.data, y.data

        def check_args(a, b, out):
            m, _n = a.shape
            n, = b.shape
            if n != _n:
                raise ValueError("incompatible array sizes for np.dot(a, b) "
                                 "(matrix * vector)")
            if out.shape != (m,):
                raise ValueError("incompatible output array size for "
                                 "np.dot(a, b, out) (matrix * vector)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    for val in m_shapes:
        check_c_int(context, builder, val)

    call_xxgemv(context, builder, do_trans, mty, m_shapes, m_data,
                v_data, out.data)

    return impl_ret_borrowed(context, builder, sig.return_type,
                             out._getvalue())


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

    # The only case Numpy supports
    assert outty.layout == 'C'

    def check_args(a, b, out):
        m, k = a.shape
        _k, n = b.shape
        if k != _k:
            raise ValueError("incompatible array sizes for np.dot(a, b) "
                             "(matrix * matrix)")
        if out.shape != (m, n):
            raise ValueError("incompatible output array size for "
                             "np.dot(a, b, out) (matrix * matrix)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    check_c_int(context, builder, m)
    check_c_int(context, builder, k)
    check_c_int(context, builder, n)

    x_data = x.data
    y_data = y.data
    out_data = out.data

    # Check whether any of the operands is really a 1-d vector represented
    # as a (1, k) or (k, 1) 2-d array.  In those cases, it is pessimal
    # to call the generic matrix * matrix product BLAS function.
    one = ir.Constant(intp_t, 1)
    is_left_vec = builder.icmp_signed('==', m, one)
    is_right_vec = builder.icmp_signed('==', n, one)

    with builder.if_else(is_right_vec) as (r_vec, r_mat):
        with r_vec:
            with builder.if_else(is_left_vec) as (v_v, m_v):
                with v_v:
                    # V * V
                    call_xxdot(context, builder, False, dtype,
                               k, x_data, y_data, out_data)
                with m_v:
                    # M * V
                    do_trans = xty.layout == outty.layout
                    call_xxgemv(context, builder, do_trans,
                                xty, x_shapes, x_data, y_data, out_data)
        with r_mat:
            with builder.if_else(is_left_vec) as (v_m, m_m):
                with v_m:
                    # V * M
                    do_trans = yty.layout != outty.layout
                    call_xxgemv(context, builder, do_trans,
                                yty, y_shapes, y_data, x_data, out_data)
                with m_m:
                    # M * M
                    call_xxgemm(context, builder,
                                xty, x_shapes, x_data,
                                yty, y_shapes, y_data,
                                outty, out_shapes, out_data)

    return impl_ret_borrowed(context, builder, sig.return_type,
                             out._getvalue())


@lower_builtin(numpy.dot, types.Array, types.Array,
               types.Array)
def dot_3(context, builder, sig, args):
    """
    np.dot(a, b, out)
    """
    ensure_blas()

    with make_contiguous(context, builder, sig, args) as (sig, args):
        ndims = set(x.ndim for x in sig.args[:2])
        if ndims == set([2]):
            return dot_3_mm(context, builder, sig, args)
        elif ndims == set([1, 2]):
            return dot_3_vm(context, builder, sig, args)
        else:
            assert 0


def call_xxgetrf(context, builder, a_type, a_shapes, a_data, ipiv, info):
    """
    Call the LAPACK gettrf function for the given argument.

    This function computes the LU decomposition of a matrix.

    """
    fnty = ir.FunctionType(ll_intc,
                           [ll_char,                       # kind
                            intp_t, intp_t,                # m, n
                            ll_void_p, intp_t,             # a, lda
                            ll_intc_p, ll_intc_p           # ipiv, info
                            ])

    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgetrf")

    kind = get_blas_kind(a_type.dtype)
    kind_val = ir.Constant(ll_char, ord(kind))

    if a_type.layout == 'F':
        m, n = a_shapes
        lda = a_shapes[0]
    else:
        n, m = a_shapes
        lda = a_shapes[1]

    res = builder.call(fn, (kind_val, m, n,
                            builder.bitcast(a_data, ll_void_p), lda,
                            ipiv, info
                            ))
    check_lapack_return(context, builder, res)


def call_xxgetri(context, builder, a_type, a_shapes, a_data, ipiv, work,
                 lwork, info):
    """
    Call the LAPACK gettri function for the given argument.

    This function computes the inverse of a matrix given its LU decomposition.

    """
    fnty = ir.FunctionType(ll_intc,
                           [ll_char,                       # kind
                            intp_t, ll_void_p, intp_t,     # n, a, lda
                            ll_intc_p, ll_void_p,          # ipiv, work
                            ll_intc_p, ll_intc_p           # lwork, info
                            ])
    fn = builder.module.get_or_insert_function(fnty, name="numba_xxgetri")

    kind = get_blas_kind(a_type.dtype)
    kind_val = ir.Constant(ll_char, ord(kind))

    n = lda = a_shapes[0]

    res = builder.call(fn, (kind_val, n,
                            builder.bitcast(a_data, ll_void_p), lda,
                            ipiv, builder.bitcast(work, ll_void_p),
                            lwork, info
                            ))
    check_lapack_return(context, builder, res)


def mat_inv(context, builder, sig, args):
    """
    Invert a matrix through the use of its LU decomposition.
    """
    xty = sig.args[0]
    dtype = xty.dtype

    x = make_array(xty)(context, builder, args[0])
    x_shapes = cgutils.unpack_tuple(builder, x.shape)
    m, n = x_shapes
    check_c_int(context, builder, m)
    check_c_int(context, builder, n)

    # Allocate the return array (Numpy never works in place contrary to
    # Scipy for which one can specify to whether or not to overwrite the
    # input).
    def create_out(a):
        m, n = a.shape
        if m != n:
            raise numpy.linalg.LinAlgError("Last 2 dimensions of "
                                           "the array must be square.")
        return a.copy()

    out = context.compile_internal(builder, create_out,
                                   signature(sig.return_type, *sig.args), args)
    o = make_array(xty)(context, builder, out)

    # Allocate the array in which the pivot indices are stored.
    ipiv_t = types.Array(types.intc, 1, 'C')
    i = _empty_nd_impl(context, builder, ipiv_t, (m,))
    ipiv = i._getvalue()

    info = cgutils.alloca_once(builder, ll_intc)

    # Compute the LU decomposition of the matrix.
    call_xxgetrf(context, builder, xty, x_shapes, o.data, i.data,
                 info)

    zero = ir.Constant(ll_intc, 0)
    info_val = builder.load(info)
    lapack_error = builder.icmp_signed('!=', info_val, zero)
    invalid_arg = builder.icmp_signed('<', info_val, zero)

    with builder.if_then(lapack_error, False):
        context.nrt_decref(builder, ipiv_t, ipiv)
        with builder.if_else(invalid_arg) as (then, otherwise):
            raise_err = context.call_conv.return_user_exc
            with then:
                raise_err(builder, ValueError,
                          ('One argument passed to getrf is invalid',)
                          )
            with otherwise:
                raise_err(builder, ValueError,
                          ('Matrix is singular and cannot be inverted',)
                          )

    # Compute the optimal lwork.
    lwork = make_constant_slot(context, builder, types.intc, -1)
    work = cgutils.alloca_once(builder, context.get_value_type(xty.dtype))
    call_xxgetri(context, builder, xty, x_shapes, o.data, i.data, work,
                 lwork, info)

    info_val = builder.load(info)
    lapack_error = builder.icmp_signed('!=', info_val, zero)

    with builder.if_then(lapack_error, False):
        context.nrt_decref(builder, ipiv_t, ipiv)
        raise_err = context.call_conv.return_user_exc
        raise_err(builder, ValueError,
                  ('One argument passed to getri is invalid',)
                  )

    # Allocate a work array of the optimal size as computed by getri.
    def allocate_work(x, size):
        """Allocate the work array.

        """
        size = int(1.01 * size.real)
        return numpy.empty((size,), dtype=x.dtype)

    wty = types.Array(dtype, 1, 'C')
    work = context.compile_internal(builder, allocate_work,
                                    signature(wty, xty, dtype),
                                    (args[0], builder.load(work)))

    w = make_array(wty)(context, builder, work)
    w_shapes = cgutils.unpack_tuple(builder, w.shape)
    lw, = w_shapes

    builder.store(context.cast(builder, lw, types.intp, types.intc),
                  lwork)

    # Compute the matrix inverse.
    call_xxgetri(context, builder, xty, x_shapes, o.data, i.data, w.data,
                 lwork, info)

    info_val = builder.load(info)
    lapack_error = builder.icmp_signed('!=', info_val, zero)
    invalid_arg = builder.icmp_signed('<', info_val, zero)

    context.nrt_decref(builder, wty, work)
    context.nrt_decref(builder, ipiv_t, ipiv)

    with builder.if_then(lapack_error, False):
        with builder.if_else(invalid_arg) as (then, otherwise):
            raise_err = context.call_conv.return_user_exc
            with then:
                raise_err(builder, ValueError,
                          ('One argument passed to getri is invalid',)
                          )
            with otherwise:
                raise_err(builder, ValueError,
                          ('Matrix is singular and cannot be inverted',)
                          )

    return impl_ret_new_ref(context, builder, sig.return_type, out)


@lower_builtin(numpy.linalg.inv, types.Array)
def inv(context, builder, sig, args):
    """
    np.linalg.inv(a)
    """
    ensure_lapack()

    ndims = sig.args[0].ndim
    if ndims == 2:
        return mat_inv(context, builder, sig, args)
    else:
        assert 0

if numpy_version >= (1, 8):

    @jit
    def _check_finite_matrix(a):  
        for v in numpy.nditer(a):
            if not numpy.isfinite(v.item()):
                raise numpy.linalg.LinAlgError(
                    "Array must not contain infs or NaNs.")

    
    def _check_linalg_matrix(a, func_name):
        if not isinstance(a, types.Array):
            raise TypingError("np.linalg.%s() only supported for array types"
                              % func_name)
        if not a.ndim == 2:
            raise TypingError("np.linalg.%s() only supported on 2-D arrays."
                              % func_name)
        if not isinstance(a.dtype, (types.Float, types.Complex)):
            raise TypingError("np.linalg.%s() only supported on "
                              "float and complex arrays." % func_name)

    @overload(numpy.linalg.cholesky)
    def cho_impl(a):
        ensure_lapack()

        _check_linalg_matrix(a, "cholesky")

        xxpotrf_sig = types.intc(types.int8, types.int8, types.intp,
                                 types.CPointer(a.dtype), types.intp,
                                 types.CPointer(types.intc))
        xxpotrf = types.ExternalFunction("numba_xxpotrf", xxpotrf_sig)

        kind = ord(get_blas_kind(a.dtype, "cholesky"))
        UP = ord('U')
        LO = ord('L')

        def cho_impl(a):
            n = a.shape[-1]
            if a.shape[-2] != n:
                msg = "Last 2 dimensions of the array must be square."
                raise numpy.linalg.LinAlgError(msg)

            # The output is allocated in C order
            out = a.copy()
            # XXX spurious heap allocation
            info = numpy.zeros(1, dtype=numpy.intc)
            # Pass UP since xxpotrf() operates in F order
            # The semantics ensure this works fine
            # (out is really its Hermitian in F order, but UP instructs
            #  xxpotrf to compute the Hermitian of the upper triangle
            #  => they cancel each other)
            r = xxpotrf(kind, UP, n, out.ctypes, n, info.ctypes)
            if r != 0:
                # XXX Py_FatalError()?
                raise RuntimeError("unable to execute xxpotrf()")
            if info[0] > 0:
                raise numpy.linalg.LinAlgError(
                    "Matrix is not positive definite.")
            elif info[0] < 0:
                # Invalid parameter
                raise RuntimeError(
                    "cholesky(): Internal error: please report an issue")
            # Zero out upper triangle, in F order
            for col in range(n):
                out[:col, col] = 0
            return out

        return cho_impl

    @overload(numpy.linalg.eig)
    def eig_impl(a):
        ensure_lapack()

        _check_linalg_matrix(a, "eig")

        numba_ez_rgeev_sig = types.intc(types.char,  # kind
                                        types.char,  # jobvl
                                        types.char,  # jobvr
                                        types.intp,  # n
                                        types.CPointer(a.dtype),  # a
                                        types.intp,  # lda
                                        types.CPointer(a.dtype),  # wr
                                        types.CPointer(a.dtype),  # wi
                                        types.CPointer(a.dtype),  # vl
                                        types.intp,  # ldvl
                                        types.CPointer(a.dtype),  # vr
                                        types.intp  # ldvr
                                        )

        numba_ez_rgeev = types.ExternalFunction("numba_ez_rgeev",
                                                numba_ez_rgeev_sig)

        numba_ez_cgeev_sig = types.intc(types.char,  # kind
                                        types.char,  # jobvl
                                        types.char,  # jobvr
                                        types.intp,  # n
                                        types.CPointer(a.dtype),  # a
                                        types.intp,  # lda
                                        types.CPointer(a.dtype),  # w
                                        types.CPointer(a.dtype),  # vl
                                        types.intp,  # ldvl
                                        types.CPointer(a.dtype),  # vr
                                        types.intp  # ldvr
                                        )

        numba_ez_cgeev = types.ExternalFunction("numba_ez_cgeev",
                                                numba_ez_cgeev_sig)

        kind = ord(get_blas_kind(a.dtype, "eig"))

        JOBVL = ord('N')
        JOBVR = ord('V')

        F_layout = a.layout == 'F'

        def real_eig_impl(a):
            n = a.shape[-1]
            if a.shape[-2] != n:
                msg = "Last 2 dimensions of the array must be square."
                raise numpy.linalg.LinAlgError(msg)

            _check_finite_matrix(a)

            if F_layout:
                acpy = numpy.copy(a)
            else:
                acpy = numpy.asfortranarray(a)

            ldvl = 1
            ldvr = n
            wr = numpy.empty(n, dtype=a.dtype)
            wi = numpy.empty(n, dtype=a.dtype)
            vl = numpy.empty((ldvl, n), dtype=a.dtype)
            vr = numpy.empty((ldvr, n), dtype=a.dtype)

            r = numba_ez_rgeev(kind,
                               JOBVL,
                               JOBVR,
                               n,
                               acpy.ctypes,
                               n,
                               wr.ctypes,
                               wi.ctypes,
                               vl.ctypes,
                               ldvl,
                               vr.ctypes,
                               ldvr)

            # By design numba does not support dynamic return types, however, 
            # Numpy does. Numpy uses this ability in the case of returning 
            # eigenvalues/vectors of a real matrix. The return type of 
            # np.linalg.eig(), when operating on a matrix in real space 
            # depends on the values present in the matrix itself (recalling 
            # that eigenvalues are the roots of the characteristic polynomial 
            # of the system matrix, which will by construction depend on the 
            # values present in the system matrix). As numba cannot handle 
            # the case of a runtime decision based domain change relative to 
            # the input type, if it is required numba raises as below.
            if numpy.any(wi):
                raise ValueError(
                    "eig() argument must not cause a domain change.")

            # put these in to help with liveness analysis,
            # `.ctypes` doesn't keep the vars alive
            acpy.size
            vl.size
            vr.size
            wr.size
            wi.size
            return (wr, vr.T)

        def cmplx_eig_impl(a):
            n = a.shape[-1]
            if a.shape[-2] != n:
                msg = "Last 2 dimensions of the array must be square."
                raise numpy.linalg.LinAlgError(msg)

            _check_finite_matrix(a)

            if F_layout:
                acpy = numpy.copy(a)
            else:
                acpy = numpy.asfortranarray(a)

            ldvl = 1
            ldvr = n
            w = numpy.empty(n, dtype=a.dtype)
            vl = numpy.empty((ldvl, n), dtype=a.dtype)
            vr = numpy.empty((ldvr, n), dtype=a.dtype)

            r = numba_ez_cgeev(kind,
                               JOBVL,
                               JOBVR,
                               n,
                               acpy.ctypes,
                               n,
                               w.ctypes,
                               vl.ctypes,
                               ldvl,
                               vr.ctypes,
                               ldvr)

            # put these in to help with liveness analysis,
            # `.ctypes` doesn't keep the vars alive
            acpy.size
            vl.size
            vr.size
            w.size
            return (w, vr.T)

        if isinstance(a.dtype, types.scalars.Complex):
            return cmplx_eig_impl
        else:
            return real_eig_impl

    @overload(numpy.linalg.svd)
    def svd_impl(a, full_matrices=1):
        ensure_lapack()

        _check_linalg_matrix(a, "svd")

        F_layout = a.layout == 'F'

        # convert typing floats to numpy floats for use in the impl
        s_type = getattr(a.dtype, "underlying_float", a.dtype)
        if s_type.bitwidth == 32:
            s_dtype = numpy.float32
        else:
            s_dtype = numpy.float64

        numba_ez_gesdd_sig = types.intc(
            types.char,  # kind
            types.char,  # jobz
            types.intp,  # m
            types.intp,  # n
            types.CPointer(a.dtype),  # a
            types.intp,  # lda
            types.CPointer(s_type),  # s
            types.CPointer(a.dtype),  # u
            types.intp,  # ldu
            types.CPointer(a.dtype),  # vt
            types.intp  # ldvt
        )

        numba_ez_gesdd = types.ExternalFunction("numba_ez_gesdd",
                                                numba_ez_gesdd_sig)

        kind = ord(get_blas_kind(a.dtype, "svd"))

        JOBZ_A = ord('A')
        JOBZ_S = ord('S')

        def svd_impl(a, full_matrices=1):
            n = a.shape[-1]
            m = a.shape[-2]

            _check_finite_matrix(a)

            if F_layout:
                acpy = numpy.copy(a)
            else:
                acpy = numpy.asfortranarray(a)

            ldu = m
            minmn = min(m, n)

            if full_matrices:
                JOBZ = JOBZ_A
                ucol = m
                ldvt = n
            else:
                JOBZ = JOBZ_S
                ucol = minmn
                ldvt = minmn

            u = numpy.empty((ucol, ldu), dtype=a.dtype)
            s = numpy.empty(minmn, dtype=s_dtype)
            vt = numpy.empty((n, ldvt), dtype=a.dtype)

            r = numba_ez_gesdd(
                kind,  # kind
                JOBZ,  # jobz
                m,  # m
                n,  # n
                acpy.ctypes,  # a
                m,  # lda
                s.ctypes,  # s
                u.ctypes,  # u
                ldu,  # ldu
                vt.ctypes,  # vt
                ldvt          # ldvt
            )

            # help liveness analysis
            acpy.size
            vt.size
            u.size
            s.size

            # TODO: this logic, no point in churning out a copy if no need
            # if not issymmetrical(u):
            # u = u.T
            # if not issymmetrical(v):
            # vt = vt.T

            return (u.T, s, vt.T)

        return svd_impl
