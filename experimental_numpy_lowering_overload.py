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
    print(type_names)
    ret = dpnp_glue.get_dpnp_fn_ptr(fn_name, type_names)
    import pdb
    pdb.set_trace()
    return

    import ctypes
    dpnp_lib = ctypes.cdll.LoadLibrary("libdpnp_backend_c.so")
    C_get_function = dpnp_lib.get_backend_function_name
    dpnp_lib.get_backend_function_name.argtype = [ctypes.c_char_p, ctypes.c_char_p]
    dpnp_lib.get_backend_function_name.restype = ctypes.c_long
    f_ptr = dpnp_lib.get_backend_function_name(fn_name, type_names[0])

    print(hex(f_ptr))

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


@lower_builtin(np.dot, types.Array, types.Array)
def dot_dppl(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """

    with make_contiguous(context, builder, sig, args) as (sig, args):
        ndims = [x.ndim for x in sig.args[:2]]
        if ndims == [2, 2]:
            print("gemm")
            #return dot_2_mm(context, builder, sig, args)
        elif ndims == [2, 1]:
            print("gemv")
            #return dot_2_mv(context, builder, sig, args)
        elif ndims == [1, 2]:
            print("gemv")
            #return dot_2_vm(context, builder, sig, args)
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


@lower_builtin(np.argsort, types.Array)
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
    def make_res(A):
        return np.empty((A.size, A.size))

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])

    if a.shape == 2:
        m, n = cgutils.unpack_tuple(builder, a.shape)
    elif a.shape == 1:
        m, = cgutils.unpack_tuple(builder, a.shape)
    else:
        #TODO: Throw error, cov is supported for only 1D and 2D array
        pass

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args[:1]), args[:1])

    outary = make_array(sig.return_type)(context, builder, out)

    import pdb
    pdb.set_trace()

    if a.shape == 2:
        shape =  cgutils.alloca_once(builder, context.get_value_type(ir.IntType(64)), size=2)
        builder.store(m, cgutils.gep(builder, shape, 0))
        builder.store(n, cgutils.gep(builder, shape, 0))
    elif a.shape == 1:
        shape =  cgutils.alloca_once(builder, context.get_value_type(ir.IntType(64)), size=1)
        builder.store(m, cgutils.gep(builder, shape, 0))

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ll_intp_p]
    params = (builder.bitcast(a.data, ll_void_p), builder.bitcast(outary.data, ll_void_p),
            builder.bitcast(shape, ll_intp_p))

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_cov", type_names, params, param_tys, ll_void)
    return out
