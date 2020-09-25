import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (lower_builtin)
from numba.core.typing import signature
from numba.np.arrayobj import make_array, _empty_nd_impl, array_copy
from numba.core import itanium_mangler
from llvmlite import ir
import contextlib

from numba import int32, int64, uint32, uint64, float32, float64


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
        context.nrt.decref(builder, ty, val)

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


ll_char = ir.IntType(8)
ll_char_p = ll_char.as_pointer()
ll_void = ir.VoidType()
ll_void_p = ll_char_p
ll_intc = ir.IntType(32)
ll_intc_p = ll_intc.as_pointer()
intp_t = cgutils.intp_t
ll_intp_p = intp_t.as_pointer()


def call_dpnp(context, builder, fn_name, type_names, params, param_tys, ret_ty):
    from .dpnp_glue import dpnp_fptr_interface as dpnp_glue
    f_ptr = dpnp_glue.get_dpnp_fn_ptr(fn_name, type_names)

    '''
    import ctypes
    dpnp_lib = ctypes.cdll.LoadLibrary("libdpnp_backend_c.so")
    C_get_function = dpnp_lib.get_backend_function_name
    dpnp_lib.get_backend_function_name.argtype = [ctypes.c_char_p, ctypes.c_char_p]
    dpnp_lib.get_backend_function_name.restype = ctypes.c_long
    f_ptr = dpnp_lib.get_backend_function_name(fn_name, type_names[0])

    print(hex(f_ptr))
    '''

    fnty = ir.FunctionType(ret_ty, param_tys)
    addr_constant = context.get_constant(int64, f_ptr)
    fn_ptr = builder.inttoptr(addr_constant, fnty.as_pointer())

    res = builder.call(fn_ptr, params)


def dot_2_vv(context, builder, sig, args, conjugate=False):
    """
    np.dot(vector, vector)
    np.vdot(vector, vector)
    """

    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))
    size, = cgutils.unpack_tuple(builder, a.shape)

    def check_args(a, b):
        m, = a.shape
        n, = b.shape
        if m != n:
            raise ValueError("incompatible array sizes for np.dot(a, b) "
                             "(vector * vector)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)
    check_c_int(context, builder, size)

    # arguments are : a->void*, b->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(b.data, ll_void_p),
              builder.bitcast(out, ll_void_p), size)

    type_names = []
    for argty in sig.args:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_dot", type_names, params, param_tys, ll_void)

    return builder.load(out)


def dot_2_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
    def make_res(a, b):
        m, n = a.shape
        _n, k = b.shape
        if _n != n:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((m, k), a.dtype)

    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m, n = cgutils.unpack_tuple(builder, a.shape)
    _n, k = cgutils.unpack_tuple(builder, b.shape)


    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64), ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p),
              builder.bitcast(b.data, ll_void_p),
              builder.bitcast(outary.data, ll_void_p),
              m, n, k)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name.encode('utf-8'))
    type_names.append(sig.return_type.name.encode('utf-8'))

    call_dpnp(context, builder, b"dpnp_matmul", type_names, params, param_tys, ll_void)
    return out


def dot_2_mv(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
    def make_res(a, b):
        m, n = a.shape
        _n = b.shape
        if _n != n:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((m, ) a.dtype)

    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m, n = cgutils.unpack_tuple(builder, a.shape)
    _n,  = cgutils.unpack_tuple(builder, b.shape)

    k = context.get_constant(types.int64, 1)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64), ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p),
              builder.bitcast(b.data, ll_void_p),
              builder.bitcast(outary.data, ll_void_p),
              m, n, k)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name.encode('utf-8'))
    type_names.append(sig.return_type.name.encode('utf-8'))

    call_dpnp(context, builder, b"dpnp_matmul", type_names, params, param_tys, ll_void)
    return out


def dot_2_vm(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
    def make_res(a, b):
        m,  = a.shape
        n, k = b.shape
        if m != n:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((k, ) a.dtype)

    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m,  = cgutils.unpack_tuple(builder, a.shape)
    n, k  = cgutils.unpack_tuple(builder, b.shape)

    m = context.get_constant(types.int64, 1)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p),
              builder.bitcast(b.data, ll_void_p),
              builder.bitcast(outary.data, ll_void_p),
              m, n, k)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name.encode('utf-8'))
    type_names.append(sig.return_type.name.encode('utf-8'))

    call_dpnp(context, builder, b"dpnp_matmul", type_names, params, param_tys, ll_void)
    return out


@lower_builtin(np.dot, types.Array, types.Array)
def dot_dppl(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """

    with make_contiguous(context, builder, sig, args) as (sig, args):
        ndims = [x.ndim for x in sig.args[:2]]
        if ndims == [2, 2]:
            return dot_2_mm(context, builder, sig, args)
        elif ndims == [2, 1]:
            return dot_2_mv(context, builder, sig, args)
        elif ndims == [1, 2]:
            return dot_2_vm(context, builder, sig, args)
        elif ndims == [1, 1]:
            print("dot")
            return dot_2_vv(context, builder, sig, args)
        else:
            assert 0
    raise ImportError("scipy 0.16+ is required for linear algebra")


@lower_builtin(np.sum, types.Array)
def array_sum(context, builder, sig, args):
    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size, = cgutils.unpack_tuple(builder, a.shape)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(out, ll_void_p), size)

    type_names = []
    for argty in sig.args:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_sum", type_names, params, param_tys, ll_void)
    return builder.load(out)


@lower_builtin(np.argmax, types.Array)
def array_argmax(context, builder, sig, args):
    def argmax_checker(arry):
        if arry.size == 0:
            raise ValueError("attempt to get argmax of an empty sequence")

    context.compile_internal(builder, argmax_checker,
                             signature(types.none, *sig.args), args)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size, = cgutils.unpack_tuple(builder, a.shape)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(out, ll_void_p), size)

    type_names = []
    for argty in sig.args:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_argmax", type_names, params, param_tys, ll_void)

    return builder.load(out)


@lower_builtin(np.argmin, types.Array)
def array_argmin(context, builder, sig, args):
    def argmin_checker(arry):
        if arry.size == 0:
            raise ValueError("attempt to get argmin of an empty sequence")

    context.compile_internal(builder, argmax_checker,
                             signature(types.none, *sig.args), args)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size, = cgutils.unpack_tuple(builder, a.shape)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(out, ll_void_p), size)

    type_names = []
    for argty in sig.args:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_argmin", type_names, params, param_tys, ll_void)

    return builder.load(out)



@lower_builtin(np.argsort, types.Array, types.StringLiteral)
def array_argsort(context, builder, sig, args):
    def make_res(A):
        return np.arange(A.size)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size, = cgutils.unpack_tuple(builder, a.shape)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args[:1]), args[:1])

    outary = make_array(sig.return_type)(context, builder, out)

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(outary.data, ll_void_p), size)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_argsort", type_names, params, param_tys, ll_void)
    return out


@lower_builtin(np.cov, types.Array)
def array_cov(context, builder, sig, args):
    def make_1D_res(size):
        return np.empty(1, dtype=np.float64)

    def make_2D_res(size):
        return np.empty((size, size), dtype=np.float64)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])

    if aty.ndim == 2:
        m, n = cgutils.unpack_tuple(builder, a.shape)
        out = context.compile_internal(builder, make_2D_res,
                signature(sig.return_type, types.int64), (m,))
    elif aty.ndim == 1:
        m, = cgutils.unpack_tuple(builder, a.shape)
        out = context.compile_internal(builder, make_1D_res,
                signature(sig.return_type, types.int64), (m,))
    else:
        #TODO: Throw error, cov is supported for only 1D and 2D array
        pass

    outary = make_array(sig.return_type)(context, builder, out)

    nrows = cgutils.alloca_once(builder, context.get_value_type(types.int64))
    ncols = cgutils.alloca_once(builder, context.get_value_type(types.int64))

    if aty.ndim == 2:
        builder.store(m, nrows)
        builder.store(n, ncols)

    elif aty.ndim == 1:
        builder.store(context.get_constant(types.int64, 1), nrows)
        builder.store(m, ncols)


    # arguments are : a ->void*, result->void*, nrows->int64, ncols->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64)]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(outary.data, ll_void_p),
              nrows, ncols)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_cov", type_names, params, param_tys, ll_void)
    return out


'''
@lower_builtin(np.linalg.eig, types.Array)
def array_cov(context, builder, sig, args):
    pass

@lower_builtin("np.random.sample")
def random_impl(context, builder, sig, args):

    def make_res(shape):
        return np.empty(shape, dtype=np.float64)

    import pdb
    pdb.set_trace()
    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    # arguments are : result->void*, size->int64
    param_tys = [ll_void_p, ll_intp_p]
    params = (builder.bitcast(outary.data, ll_void_p), )


    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name.encode('utf-8'))
    type_names.append(sig.return_type.name.encode('utf-8'))

    call_dpnp(context, builder, b"dpnp_cov", type_names, params, param_tys, ll_void)
'''
