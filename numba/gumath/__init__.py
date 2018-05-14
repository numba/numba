from gumath import unsafe_add_kernel
from .. import jit
from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct
from toolz.functoolz import thread_first as tf

from .xnd_types import *

def jit_to_kernel(gumath_sig, numba_sig):
    """
    JIT compiles a function and returns a 0D XND kernel for it.
    Call with the ndtype function signature and the numba signature of the inner function.
    >>> import math
    >>> import numpy as np
    >>> from xnd import xnd
    >>> @jit_to_kernel('... * float64 -> ... * float64', 'float64(float64)')
    ... def f(x):
    ...     return math.sin(x)
    <_gumath.gufunc at 0x10e3d5f90>
    >>> f(xnd.from_buffer(np.arange(20).astype('float64')))
    xnd([0.0,
         0.8414709848078965,
         0.9092974268256817,
         0.1411200080598672,
         -0.7568024953079282,
         -0.9589242746631385,
         -0.27941549819892586,
         0.6569865987187891,
         0.9893582466233818,
         ...],
        type='20 * float64')
    """
    return lambda fn: _gu_vectorize(fn, gumath_sig, numba_sig)

# global counter for gumath kernel functions

i = 0

def _gu_vectorize(fn, gumath_sig, numba_sig):
    global i
    # JIT right now by passing in numba signature
    dispatcher = jit(numba_sig, nopython=True)(fn)
    # grab the first jitted function, since we are only doing one
    cres = list(dispatcher.overloads.values())[0]
    llvm_name = cres.fndesc.llvm_func_name

    ctx = cres.target_context
    func_ptr = build_kernel_wrapper(
        library=cres.library,
        context=ctx,
        fname=llvm_name,
        signature=cres.signature,
        envptr=cres.environment.as_pointer(ctx)
    )

    # gumath kernel name needs to be unique
    kernel_name = 'numba' + str(i)
    i += 1
    return unsafe_add_kernel(
        name=kernel_name,
        sig=gumath_sig,
        ptr=func_ptr,
        vectorize=False,
        tag='Xnd'
    )



def build_kernel_wrapper(library, context, fname, signature, envptr):
    """
    Returns a pointer to a llvm function that can be used as an xnd kernel.
    Like build_ufunc_wrapper
    """

    # setup the module and jitted function
    wrapperlib = context.codegen().create_library('gumath_wrapper')
    wrapper_module = wrapperlib.create_ir_module('')

    func_type = context.call_conv.get_function_type(
        signature.return_type, signature.args)
    func = wrapper_module.add_function(func_type, name=fname)
    func.attributes.add("alwaysinline")

    # create 0D xnd kernel function
    # like https://github.com/plures/gumath/blob/2eba3ba8cc7d4828232138f9805bcd9bb99248ae/libgumath/kernels.c#L135
    fnty = ir.FunctionType(
        i32,
        (
            ptr(xnd_t),
            ptr(ndt_context_t),
        )
    )

    # we will return a pointer to this function
    wrapper = wrapper_module.add_function(fnty, "__gumath__." + func.name)
    stack, ndt_context = wrapper.args
    builder = ir.IRBuilder(wrapper.append_basic_block("entry"))

    inputs = []
    for i, typ in enumerate(signature.args):
        llvm_type = context.get_data_type(typ)
        inputs.append(tf(
            stack,
            # get ith input from stack and get ptr to data from xnd object
            (builder.gep, [index(i), index(3)], True),
            (builder.bitcast, ptr(ptr(llvm_type))),
            # gep returns a pointer, so we need to load twice
            builder.load,
            builder.load
        ))


    llvm_return_type = context.get_data_type(signature.return_type)
    # pointer to output
    out = tf(
        stack,
        (builder.gep, [index(len(signature.args)), index(3)], True),
        (builder.bitcast, ptr(ptr(llvm_return_type))),
        builder.load
    )

    # called numba jitted function on inputs
    status, retval = context.call_conv.call_function(builder, func,
                                                     signature.return_type,
                                                     signature.args, inputs, env=envptr)

    with builder.if_then(status.is_error, likely=False):
        # return -1 on failure
        builder.ret(ir.Constant(i32, -1))

    builder.store(retval, out)
    # return 0 on success
    builder.ret(ir.Constant(i32, 0))

    # cleanup and return pointer
    del builder

    # print(wrapper_module)
    wrapperlib.add_ir_module(wrapper_module)
    wrapperlib.add_linking_library(library)
    return wrapperlib.get_pointer_to_function(wrapper.name)
