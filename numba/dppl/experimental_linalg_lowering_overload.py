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
ll_void_p = ll_char_p
ll_intc = ir.IntType(32)
ll_intc_p = ll_intc.as_pointer()
intp_t = cgutils.intp_t
ll_intp_p = intp_t.as_pointer()

def call_experimental_dot(context, builder, conjugate, dtype,
                          n, a_data, b_data, out_data):

    import ctypes
    inumpy_lib = ctypes.cdll.LoadLibrary("libinumpy_backend_c.so")
    C_get_function = inumpy_lib._Z25get_backend_function_namePKcS0_
    C_get_function.restype = ctypes.c_long
    C_get_function.argtype = [ctypes.c_char_p, ctypes.c_char_p]
    f_ptr = inumpy_lib.get_function("inumpy_dot", "float")
    print(f_ptr)

    fnty = ir.FunctionType(ir.IntType(32),
                           [ll_void_p, ll_void_p, ll_void_p, ir.IntType(64)])


    addr_constant = context.get_constant(int64, f_ptr)
    fn_ptr = builder.inttoptr(addr_constant, fnty.as_pointer())

    res = builder.call(fn_ptr, (builder.bitcast(a_data, ll_void_p),
                            builder.bitcast(b_data, ll_void_p),
                            builder.bitcast(out_data, ll_void_p),
                            n))


def dot_2_vv(context, builder, sig, args, conjugate=False):
    """
    np.dot(vector, vector)
    np.vdot(vector, vector)
    """
    import llvmlite.binding as ll
    ll.load_library_permanently('libinumpy.so')

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
    call_experimental_dot(context, builder, conjugate, dtype, n, a.data, b.data, out)
    return builder.load(out)


@lower_builtin(np.dot, types.Array, types.Array)
def dot_dppl(context, builder, sig, args):
    """
    np.dot(a, b)
    a @ b
    """
    import dppl.ocldrv as driver
    device = driver.runtime.get_current_device()

    # the device env should come from the context but the current context
    # is a cpu context and not a dppl_gpu_context

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
