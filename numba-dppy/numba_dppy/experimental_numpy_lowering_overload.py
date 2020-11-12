import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (lower_builtin)
from numba.core.typing import signature
from numba.np.arrayobj import make_array, _empty_nd_impl, array_copy
from numba.core import itanium_mangler
from llvmlite import ir
import llvmlite.llvmpy.core as lc
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
ll_intp_t = ir.IntType(64)
ll_intp_p = intp_t.as_pointer()


def ensure_dpnp(name):
    try:
       # import dpnp
        from .dpnp_glue import dpnp_fptr_interface as dpnp_glue
    except ImportError:
        raise ImportError("dpNP is needed to call np.%s" % name)

def get_total_size_of_array(context, builder, aty, ary):
    total_size = cgutils.alloca_once(builder, ll_intp_t)
    builder.store(builder.sext(builder.mul(ary.nitems,
       context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(aty)))), ll_intp_t), total_size)
    return builder.load(total_size)

def get_sycl_queue(context, builder):
    void_ptr_t = context.get_value_type(types.voidptr)
    get_queue_fnty = lc.Type.function(void_ptr_t, ())
    get_queue = builder.module.get_or_insert_function(get_queue_fnty,
                                            name="DPPLQueueMgr_GetCurrentQueue")
    sycl_queue_val = cgutils.alloca_once(builder, void_ptr_t)
    builder.store(builder.call(get_queue, []), sycl_queue_val)

    return sycl_queue_val

def allocate_usm(context, builder, size, sycl_queue):
    void_ptr_t = context.get_value_type(types.voidptr)
    usm_shared_fnty = lc.Type.function(void_ptr_t, [ll_intp_t, void_ptr_t])
    usm_shared = builder.module.get_or_insert_function(usm_shared_fnty,
                                                       name="DPPLmalloc_shared")

    buffer_ptr = cgutils.alloca_once(builder, void_ptr_t)
    args = [size, builder.load(sycl_queue)]
    builder.store(builder.call(usm_shared, args), buffer_ptr)

    return builder.load(buffer_ptr)

def copy_usm(context, builder, src, dst, size, sycl_queue):
    void_ptr_t = context.get_value_type(types.voidptr)
    queue_memcpy_fnty = lc.Type.function(ir.VoidType(), [void_ptr_t, void_ptr_t, void_ptr_t,
                                                         ll_intp_t])
    queue_memcpy = builder.module.get_or_insert_function(queue_memcpy_fnty,
                                                       name="DPPLQueue_Memcpy")
    args = [builder.load(sycl_queue),
            builder.bitcast(dst, void_ptr_t),
            builder.bitcast(src, void_ptr_t),
            size]
    builder.call(queue_memcpy, args)


def free_usm(context, builder, usm_buf, sycl_queue):
    void_ptr_t = context.get_value_type(types.voidptr)

    usm_free_fnty = lc.Type.function(ir.VoidType(), [void_ptr_t, void_ptr_t])
    usm_free = builder.module.get_or_insert_function(usm_free_fnty,
                                               name="DPPLfree_with_queue")

    builder.call(usm_free, [usm_buf, builder.load(sycl_queue)])


def call_dpnp(context, builder, fn_name, type_names, params, param_tys, ret_ty):
    from .dpnp_glue import dpnp_fptr_interface as dpnp_glue
    f_ptr = dpnp_glue.get_dpnp_fn_ptr(fn_name, type_names)

    fnty = ir.FunctionType(ret_ty, param_tys)
    addr_constant = context.get_constant(int64, f_ptr)
    fn_ptr = builder.inttoptr(addr_constant, fnty.as_pointer())

    res = builder.call(fn_ptr, params)


def dot_2_vv(context, builder, sig, args, conjugate=False):
    """
    np.dot(vector, vector)
    np.vdot(vector, vector)
    """
    def check_args(a, b):
        m, = a.shape
        n, = b.shape
        if m != n:
            print("SIZES ", m, n)
            raise ValueError("incompatible array sizes for np.dot(a, b) "
                             "(vector * vector)")

    context.compile_internal(builder, check_args,
                             signature(types.none, *sig.args), args)


    sycl_queue = get_sycl_queue(context, builder)
    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    size, = cgutils.unpack_tuple(builder, a.shape)

    check_c_int(context, builder, size)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    total_size_b = get_total_size_of_array(context, builder, bty.dtype, b)
    b_usm = allocate_usm(context, builder, total_size_b, sycl_queue)
    copy_usm(context, builder, b.data, b_usm, total_size_b, sycl_queue)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))
    builder.store(context.get_constant(sig.return_type, 0), out)
    out_usm = allocate_usm(context, builder,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    # arguments are : a->void*, b->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64)]
    params = (a_usm, b_usm, out_usm, size)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_dot", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, out,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

    return builder.load(out)


def dot_2_mv(context, builder, sig, args):
    """
    np.dot(matrix, matrix)
    """
    def make_res(a, b):
        m, n = a.shape
        _n, = b.shape
        if _n != n:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((m, ), a.dtype)

    sycl_queue = get_sycl_queue(context, builder)

    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m, k = cgutils.unpack_tuple(builder, a.shape)
    _n,  = cgutils.unpack_tuple(builder, b.shape)
    n = context.get_constant(types.int64, 1)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    total_size_b = get_total_size_of_array(context, builder, bty.dtype, b)
    b_usm = allocate_usm(context, builder, total_size_b, sycl_queue)
    copy_usm(context, builder, b.data, b_usm, total_size_b, sycl_queue)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    total_size_out = get_total_size_of_array(context, builder, sig.return_type.dtype, outary)
    out_usm = allocate_usm(context, builder, total_size_out, sycl_queue)

    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64), ir.IntType(64)]
    params = (a_usm, b_usm, out_usm, m, n, k)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_matmul", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, outary.data, total_size_out, sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, b_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)
    return out


def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """
    def make_res(a, b):
        m,  = a.shape
        _m, n = b.shape
        if m != _m:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((n, ), a.dtype)

    sycl_queue = get_sycl_queue(context, builder)
    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m,  = cgutils.unpack_tuple(builder, a.shape)
    k, n  = cgutils.unpack_tuple(builder, b.shape)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    total_size_b = get_total_size_of_array(context, builder, bty.dtype, b)
    b_usm = allocate_usm(context, builder, total_size_b, sycl_queue)
    copy_usm(context, builder, b.data, b_usm, total_size_b, sycl_queue)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)

    total_size_out = get_total_size_of_array(context, builder, sig.return_type.dtype, outary)
    out_usm = allocate_usm(context, builder, total_size_out, sycl_queue)


    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64), ir.IntType(64)]
    params = (a_usm, b_usm, out_usm, m, n, k)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_matmul", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, outary.data, total_size_out, sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, b_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)
    return out


def dot_2_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix), np.matmul(matrix, matrix)
    """
    def make_res(a, b):
        m, k = a.shape
        _k, n = b.shape
        if _k != k:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        return np.empty((m, n), a.dtype)

    sycl_queue = get_sycl_queue(context, builder)
    aty, bty = sig.args
    a = make_array(aty)(context, builder, args[0])
    b = make_array(bty)(context, builder, args[1])
    m, k = cgutils.unpack_tuple(builder, a.shape)
    _k, n = cgutils.unpack_tuple(builder, b.shape)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    total_size_b = get_total_size_of_array(context, builder, bty.dtype, b)
    b_usm = allocate_usm(context, builder, total_size_b, sycl_queue)
    copy_usm(context, builder, b.data, b_usm, total_size_b, sycl_queue)


    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args), args)

    outary = make_array(sig.return_type)(context, builder, out)
    total_size_out = get_total_size_of_array(context, builder, sig.return_type.dtype, outary)
    out_usm = allocate_usm(context, builder, total_size_out, sycl_queue)


    # arguments are : a->void*, b->void*, result->void*, m->int64, n->int64, k->int64
    param_tys = [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64), ir.IntType(64), ir.IntType(64)]
    params = (a_usm,
              b_usm,
              out_usm,
              m, n, k)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_matmul", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, outary.data, total_size_out, sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, b_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)
    return out


@lower_builtin(np.dot, types.Array, types.Array)
def dot_dppl(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """

    ensure_dpnp("dot")

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
    raise ImportError("scipy 0.16+ is required for linear algebra")


@lower_builtin("np.matmul", types.Array, types.Array)
def matmul_dppl(context, builder, sig, args):
    """
    np.matmul(matrix, matrix)
    """
    ensure_dpnp("matmul")
    with make_contiguous(context, builder, sig, args) as (sig, args):
        ndims = [x.ndim for x in sig.args[:2]]
        if ndims != [2, 2]:
            raise ValueError("array dimension has to be 2 for np.matmul(a, b)")

        return dot_2_mm(context, builder, sig, args)


def common_sum_prod_impl(context, builder, sig, args, fn_type):
    def array_size_checker(arry):
        if arry.size == 0:
            raise ValueError("Passed Empty array")

    context.compile_internal(builder, array_size_checker,
                             signature(types.none, *sig.args), args)

    sycl_queue = get_sycl_queue(context, builder)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size = a.nitems

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))
    builder.store(context.get_constant(sig.return_type, 0), out)
    out_usm = allocate_usm(context, builder,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(aty.dtype))), sycl_queue)

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (a_usm, out_usm, size)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append("NONE")

    call_dpnp(context, builder, fn_type, type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, out,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(aty.dtype))), sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

    return builder.load(out)



@lower_builtin(np.sum, types.Array)
def array_sum(context, builder, sig, args):
    ensure_dpnp("sum")
    return common_sum_prod_impl(context, builder, sig, args, "dpnp_sum")


@lower_builtin(np.prod, types.Array)
def array_prod(context, builder, sig, args):
    ensure_dpnp("prod")

    return common_sum_prod_impl(context, builder, sig, args, "dpnp_prod")


def common_max_min_impl(context, builder, sig, args, fn_type):
    def array_size_checker(arry):
        if arry.size == 0:
            raise ValueError("Passed Empty array")

    context.compile_internal(builder, array_size_checker,
                             signature(types.none, *sig.args), args)

    sycl_queue = get_sycl_queue(context, builder)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    a_shape = builder.gep(args[0].operands[0], [context.get_constant(types.int32, 0), context.get_constant(types.int32, 5)])
    a_ndim = context.get_constant(types.intp, aty.ndim)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))
    builder.store(context.get_constant(sig.return_type, 0), out)
    out_usm = allocate_usm(context, builder,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    # arguments are : a ->void*, result->void*
    param_tys = [ll_void_p, ll_void_p, ll_intp_p, ir.IntType(64), ll_intp_p, ir.IntType(64)]
    params = (a_usm, out_usm, builder.bitcast(a_shape, ll_intp_p), a_ndim,
              builder.bitcast(a_shape,  ll_intp_p), a_ndim)

    type_names = []
    type_names.append(aty.dtype.name)
    if fn_type == "dpnp_mean":
        type_names.append(aty.dtype.name)
    else:
        type_names.append(sig.return_type.name)

    call_dpnp(context, builder, fn_type, type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, out,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

    return builder.load(out)


@lower_builtin(np.max, types.Array)
@lower_builtin("array.max", types.Array)
def array_max(context, builder, sig, args):
    ensure_dpnp("max")

    return common_max_min_impl(context, builder, sig, args, "dpnp_max")

@lower_builtin(np.min, types.Array)
@lower_builtin("array.min", types.Array)
def array_min(context, builder, sig, args):
    ensure_dpnp("min")

    return common_max_min_impl(context, builder, sig, args, "dpnp_min")

@lower_builtin(np.mean, types.Array)
@lower_builtin("array.mean", types.Array)
def array_mean(context, builder, sig, args):
    ensure_dpnp("mean")

    return common_max_min_impl(context, builder, sig, args, "dpnp_mean")

@lower_builtin(np.median, types.Array)
def array_median(context, builder, sig, args):
    ensure_dpnp("median")

    return common_max_min_impl(context, builder, sig, args, "dpnp_median")


def common_argmax_argmin_impl(context, builder, sig, args, fn_type):
    def array_size_checker(arry):
        if arry.size == 0:
            raise ValueError("Passed Empty array")

    context.compile_internal(builder, array_size_checker,
                             signature(types.none, *sig.args), args)

    sycl_queue = get_sycl_queue(context, builder)

    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size = a.nitems

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    out = cgutils.alloca_once(builder, context.get_value_type(sig.return_type))
    builder.store(context.get_constant(sig.return_type, 0), out)
    out_usm = allocate_usm(context, builder,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (a_usm, out_usm, size)

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, fn_type, type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, out,
            context.get_constant(types.intp, context.get_abi_sizeof(context.get_value_type(sig.return_type))), sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

    return builder.load(out)



@lower_builtin(np.argmax, types.Array)
def array_argmax(context, builder, sig, args):
    ensure_dpnp("argmax")

    return common_argmax_argmin_impl(context, builder, sig, args, "dpnp_argmax")


@lower_builtin(np.argmin, types.Array)
def array_argmin(context, builder, sig, args):
    ensure_dpnp("argmin")

    return common_argmax_argmin_impl(context, builder, sig, args, "dpnp_argmin")


@lower_builtin(np.argsort, types.Array, types.StringLiteral)
def array_argsort(context, builder, sig, args):
    ensure_dpnp("argsort")

    def make_res(A):
        return np.arange(A.size)

    def array_dim_checker(arry):
        if arry.ndim > 1:
            raise ValueError("Argsort is only supported for 1D array")

    context.compile_internal(builder, array_dim_checker,
            signature(types.none, *sig.args[:1]), args[:1])

    sycl_queue = get_sycl_queue(context, builder)
    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])
    size, = cgutils.unpack_tuple(builder, a.shape)

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

    out = context.compile_internal(builder, make_res,
            signature(sig.return_type, *sig.args[:1]), args[:1])
    outary = make_array(sig.return_type)(context, builder, out)

    total_size_out = get_total_size_of_array(context, builder, sig.return_type.dtype, outary)
    out_usm = allocate_usm(context, builder, total_size_out, sycl_queue)


    # arguments are : a ->void*, result->void*, size->int64
    param_tys = [ll_void_p, ll_void_p, ir.IntType(64)]
    params = (a_usm, out_usm, size)

    type_names = []
    for argty in sig.args[:1]:
        type_names.append(argty.dtype.name)
    type_names.append(sig.return_type.name)

    call_dpnp(context, builder, "dpnp_argsort", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, outary.data, total_size_out, sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

    return out


@lower_builtin(np.cov, types.Array)
def array_cov(context, builder, sig, args):
    ensure_dpnp("cov")
    def make_1D_res(size):
        return np.empty(1, dtype=np.float64)

    def make_2D_res(size):
        return np.empty((size, size), dtype=np.float64)

    sycl_queue = get_sycl_queue(context, builder)
    aty = sig.args[0]
    aty = sig.args[0]
    a = make_array(aty)(context, builder, args[0])

    total_size_a = get_total_size_of_array(context, builder, aty.dtype, a)
    a_usm = allocate_usm(context, builder, total_size_a, sycl_queue)
    copy_usm(context, builder, a.data, a_usm, total_size_a, sycl_queue)

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

    total_size_out = get_total_size_of_array(context, builder, sig.return_type.dtype, outary)
    out_usm = allocate_usm(context, builder, total_size_out, sycl_queue)

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
    params = (a_usm, out_usm, builder.load(nrows), builder.load(ncols))

    type_names = []
    type_names.append(aty.dtype.name)
    type_names.append("NONE")


    call_dpnp(context, builder, "dpnp_cov", type_names, params, param_tys, ll_void)

    copy_usm(context, builder, out_usm, outary.data, total_size_out, sycl_queue)

    free_usm(context, builder, a_usm, sycl_queue)
    free_usm(context, builder, out_usm, sycl_queue)

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
