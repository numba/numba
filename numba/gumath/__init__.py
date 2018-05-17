from gumath import unsafe_add_kernel
from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct
from functools import partial

from .. import jit
from .xnd_types import *


# global counter for gumath kernel functions
i = 0

def curry(f):
    return partial(partial, f)

@curry
def jit_xnd(gumath_sig, numba_sig, ndims, fn):
    """
    JIT compiles a function and returns a XND kernel for it.

    Call with the ndtype function signature, the numba signature, and the number of dimensions
    in each argument and the return value. If the return value as more than 0 dimensions,
    it will be allocated and passed into your function, so you can fill it up. In this
    case, the value you return will be ignored.
    """
    global i
    # JIT right now by passing in numba signature
    dispatcher = jit(numba_sig, nopython=True)(fn)
    # grab the first jitted function, since we are only doing one
    cres = list(dispatcher.overloads.values())[0]
    llvm_name = cres.fndesc.llvm_func_name

    ctx = cres.target_context
    func_ptr = build_kernel_wrapper(
        library=cres.library,
        ctx=ctx,
        fname=llvm_name,
        signature=cres.signature,
        envptr=cres.environment.as_pointer(ctx),
        ndims=ndims
    )

    # gumath kernel name needs to be unique
    kernel_name = 'numba' + str(i)
    i += 1
    return unsafe_add_kernel(
        name=kernel_name,
        sig=gumath_sig,
        ptr=func_ptr,
        tag='Xnd'
    )



def build_kernel_wrapper(library, ctx, fname, signature, envptr, ndims):
    """
    Returns a pointer to a llvm function that can be used as an xnd kernel.
    Like build_ufunc_wrapper
    """

    # setup the module and jitted function
    wrapperlib = ctx.codegen().create_library('gumath_wrapper')
    wrapper_module = wrapperlib.create_ir_module('')

    func_type = ctx.call_conv.get_function_type(
        signature.return_type, signature.args)
    func = wrapper_module.add_function(func_type, name=fname)
    func.attributes.add("alwaysinline")

    # create xnd kernel function
    # we will return a pointer to this function
    wrapper = wrapper_module.add_function(ir.FunctionType(
        i32,
        (
            ptr(xnd_t),
            ptr(ndt_context_t),
        )
    ), "__gumath__." + func.name)
    stack, _ = wrapper.args
    builder = ir.IRBuilder(wrapper.append_basic_block("entry"))

    return_output = ndims[-1] == 0
    in_and_outs = []
    in_and_out_types = list(signature.args)
    if return_output:
        in_and_out_types.append(signature.return_type)
    for i, typ in enumerate(in_and_out_types):
        llvm_type = ctx.get_data_type(typ)
        ndim = ndims[i]
    
        if ndim == 0:
            val = builder.load(
                builder.bitcast(
                    builder.gep(stack, [index(i), index(3)], True),
                    ptr(ptr(llvm_type))
                )
            )
            processing_return_value = return_output and i == (len(in_and_out_types) - 1)
            if not processing_return_value:
                val = builder.load(val)
            in_and_outs.append(val)
            continue
        
        t = builder.load(
            builder.gep(stack, [index(i), index(2)], True),
        )
        # transform xnd_t into a struct that numba wants when passing arrays to jitted function
        # inspured by gm_as_ndarray
        meminfo = ir.Constant(ptr(i8), None)
        parent = ir.Constant(ptr(i8), None)
        datasize = builder.load(builder.gep(t, [index(0), index(4)], True))
        item_ptr_type = llvm_type.elements[4]
        itemsize = ir.Constant(i64, ctx.get_abi_sizeof(item_ptr_type.pointee))
        nitems = builder.sdiv(datasize, itemsize)
        data = builder.load(
            builder.bitcast(
                builder.gep(stack, [index(i), index(3)], True),
                ptr(item_ptr_type)
            )
        )
        shape = builder.load(builder.alloca(ir.ArrayType(i64, ndim)))
        strides = builder.load(builder.alloca(ir.ArrayType(i64, ndim)))
        for j in range(ndim):
            # a->shape[j] = t->FixedDim.shape;
            shape = builder.insert_value(shape, builder.load(
                builder.gep(t, [index(0), index(6), index(0), index(0)], True)
            ), j)

            # a->strides[j] = t->Concrete.FixedDim.step * a->itemsize;
            strides = builder.insert_value(strides, builder.mul(itemsize, builder.load(
                builder.gep(t, [index(0), index(7), index(0), index(0), index(1)], True), 
            )), j)
            # t=t->FixedDim.type
            t = builder.load(builder.bitcast(
                builder.gep(t, [index(0), index(6), index(0), index(1)], True),
                ptr(ptr(ndt_t))
            ))
        in_and_outs.append(create_literal_struct(builder, llvm_type, [
            meminfo, parent, nitems, itemsize, data, shape, strides
        ]))

    if return_output:
        *inputs, output = in_and_outs
    else:
        inputs = in_and_outs

    # called numba jitted function on inputs
    status, retval = ctx.call_conv.call_function(builder, func,
                                                     signature.return_type,
                                                     signature.args, inputs, env=envptr)

    with builder.if_then(status.is_error, likely=False):
        # return -1 on failure
        builder.ret(ir.Constant(i32, -1))

    if return_output:
        builder.store(retval, output)
    # return 0 on success
    builder.ret(ir.Constant(i32, 0))

    wrapperlib.add_ir_module(wrapper_module)
    wrapperlib.add_linking_library(library)
    return wrapperlib.get_pointer_to_function(wrapper.name)


def create_literal_struct(builder, struct_type, values):
    """
    Takes a list of value and creates a literal struct and fill it with those fields
    """
    s = builder.load(builder.alloca(struct_type))
    for i, v in enumerate(values):
        s = builder.insert_value(s, v, i)
    return s
