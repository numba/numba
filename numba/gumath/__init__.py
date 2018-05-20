from gumath import unsafe_add_kernel
from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct
from functools import partial
from ndtypes import ndt

from .. import jit
from .. import types as numba_types

from .xnd_types import *


def jit_xnd(ndt_sig):
    return lambda fn: GumathDispatcher(ndt_sig, fn)


class GumathDispatcher:
    """
    Add gumath kernels based on Python functions.
    """
    i = 0

    def __init__(self, ndt_sig, fn):
        self.cache = {}
        self.ndt_sig = ndt_sig
        self.dimensions = list(dimensions_from_ndt(ndt_sig))
        self.returns_scalar = self.dimensions[-1] == 0
        self.dispatcher =  jit(nopython=True)(fn)
    
    def __call__(self, *args):
        dtypes = tuple(a.type.hidden_dtype for a in args)
        return self.get_or_create_kernel(dtypes)(*args)

    def get_or_create_kernel(self, dtypes):
        if dtypes in self.cache:
            return self.cache[dtypes]
        numba_sig = self.generate_numba_sig(dtypes)

        # numba gives us back the function, but we want the compile result
        # so we search for it
        entry_point = self.dispatcher.compile(numba_sig)
        cres = [cres for cres in self.dispatcher.overloads.values() if cres.entry_point == entry_point][0]

        # gumath kernel name needs to be unique
        kernel = unsafe_add_kernel(
            name= f'numba.{self.i}',
            sig=self.ndt_sig,
            ptr=build_kernel_wrapper(cres, self.dimensions),
            tag='Xnd'
        )
        self.i += 1
        self.cache[dtypes] = kernel
        return kernel

    def generate_numba_sig(self, dtypes):
        if not self.returns_scalar:
            dtypes = list(dtypes) + [infer_return_dtype(self.ndt_sig, dtypes)]
    
        return tuple(numba_argument(dtype, ndim) for dtype, ndim in zip(dtypes, self.dimensions))

def ndt_fn_to_dims(ndt_sig):
    """
    Returns the inputs and return value of the ndt signature, split by dimension

        >>> ndt_fn_to_dims("... * float64, ... * D -> D")
        (('... float64', '... D'), ('D'))
    """
    for args in ndt_sig.split(" -> "):
        yield [arg for arg in args.split(", ")]


def dimensions_from_ndt(ndt_sig):
    inputs, returns = ndt_fn_to_dims(ndt_sig)
    if len(returns) != 1:
        raise NotImplementedError("Only supports one return vale in gumath signature")
    for arg in inputs + returns:
        yield len([None for dim in arg.split(' * ') if '...' not in dim]) - 1


NDT_TO_NUMBA = {
    ndt("int64"): numba_types.int64,
    ndt("float64"): numba_types.float64
}

def numba_argument(ndt_dtype, ndim):
    numba_type = NDT_TO_NUMBA[ndt_dtype]
    if ndim == 0:
        return numba_type
    return numba_types.npytypes.Array(numba_type, ndim, 'A')

def infer_return_dtype(ndt_sig, input_dtypes):
    """
    Determines the return dtype based on the input dtypes.

        >>> infer_return_dtype('... * D, ... * K -> ... * D', ('float64', 'int32'))
        float64
    """
    inputs, returns = ndt_fn_to_dims(ndt_sig)
    *input_types, return_type = [ndt(arg).hidden_dtype for arg in inputs + returns]
    if return_type.isconcrete():
        return return_type


    for i, input_type in enumerate(input_types):
        if input_type == return_type:
            return input_dtypes[i]
    
    raise NotImplementedError(f'Cannot infer return dtype for {ndt_sig} based on inputs {input_dtypes}')


def build_kernel_wrapper(cres, ndims):
    """
    Returns a pointer to a llvm function that can be used as an xnd kernel.
    Like build_ufunc_wrapper
    """
    ctx = cres.target_context
    library = cres.library
    signature = cres.signature
    envptr = cres.environment.as_pointer(ctx)
    # setup the module and jitted function
    wrapperlib = ctx.codegen().create_library('gumath_wrapper')
    wrapper_module = wrapperlib.create_ir_module('')

    func_type = ctx.call_conv.get_function_type(
        signature.return_type, signature.args)
    func = wrapper_module.add_function(func_type, name=cres.fndesc.llvm_func_name)
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

def curry(f):
    return partial(partial, f)
